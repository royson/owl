(*
 * OWL - an OCaml numerical library for scientific computing
 * Copyright (c) 2016-2018 Liang Wang <liang.wang@cl.cam.ac.uk>
 *)

(** Neural network: interface of parallel engine *)


open Owl_algodiff.S
open Owl_optimise.S

(* module signature of model parallel engine *)

module type EngineSig = sig

  type param_context
  type barrier = ASP | PASP | BSP | SSP | PSP

  (* functions of parameter server engine *)

  val get : 'a -> 'b * int

  val set : 'a -> 'b -> unit

  val worker_num : unit -> int

  val progressive_num : unit -> int

  val start : ?barrier:barrier -> string -> string -> unit

  val register_barrier : (param_context ref -> int * (string list)) -> unit

  val register_schedule : ('a list -> ('a * ('b * ('c * 'd * 'e)) list) list) -> unit

  val register_pull : (string -> ('a * ('b * 'c)) list -> ('a * 'd) list) -> unit

  val register_push : ('a -> ('b * 'c) list -> ('b * ('d * 'e)) list) -> unit

  val register_stop : (param_context ref -> bool) -> unit

  val add_workers : int -> bool

  val remove_workers : int -> bool

end


(* module signature of neural network model *)

module type ModelSig = sig

  type network

  val mkpar : network -> t array array

  val init : network -> unit

  val update : network -> t array array -> unit

  val copy : network -> network

  val model : network -> arr -> arr

  val save : network -> string -> unit
  
  val load : string -> network

  val forward : network -> t -> t * t array array

  val calculate_gradient : ?params:Params.typ -> ?init_model:bool -> network -> t -> t -> int -> t array array * t

  val update_network : ?state:Checkpoint.state -> ?params:Params.typ -> ?init_model:bool -> network 
                      -> (t array array * t array array) -> int -> t -> int -> Checkpoint.state

end


(* implementation of parallel neural network training *)

module Make (M : ModelSig) (E : EngineSig) = struct

  type server_task = {
    mutable sid                : int;
    mutable state              : Checkpoint.state option;
    mutable server_params      : Params.typ;
    mutable model              : M.network;
    mutable train_size         : int;        (* Training data size for batch calculation *)
    mutable test_x             : t;          
    mutable test_y             : t;
    mutable val_x              : t;
    mutable val_y              : t;
    mutable start_at           : float;      (* Time training starts *)
    mutable schedule_no        : int;        (* Number of task scheduled. For Mini-Batch *)
    (* For Graph Plots *)
    mutable loss               : float list; (* Losses received *)
    mutable time               : float list; (* Total time executed by task *)
    (* For early stopping *)
    mutable lowest_val_loss      : float;
    mutable patience           : int;
  }

  type client_task = {
    mutable cid                : int;
    mutable client_params      : Params.typ;
    mutable train_x            : t;          
    mutable train_y            : t;
  }


  let make_server_task sid server_params model train_x val_x val_y test_x test_y = {
    sid;
    state = None;
    server_params;
    model;
    train_size = Arr.(shape train_x).(0);
    test_x;
    test_y;
    val_x;
    val_y;
    start_at = Unix.gettimeofday ();
    schedule_no = 0;
    loss = [];
    time = [];
    lowest_val_loss = 0.;
    patience = 0;
  }

  let make_client_task cid client_params train_x train_y = {
    cid;
    client_params;
    train_x;
    train_y;
  }
            
  (* Plot the loss function per time *)
  let plot_loss_time id losses time =
    let open Owl_plot in
    (* Might replace List with an Array with a length counter instead *)
    let losses = List.rev losses in
    let time = List.rev time in

    (* Bug: Currently crashes on Ubuntu 16.04. Writing to file instead.*)
    (* let h = create ("Time - Distributed MNist Conv2d Adam 0.01.png") in
    let x_range = float_of_int ((List.length losses)) in
    let y_range l = List.fold_left Pervasives.max (List.hd l) l |> Pervasives.(+.) in
    Owl_log.debug "Plotting loss function.. ";
    set_foreground_color h 0 0 0;
    set_background_color h 255 255 255;
    set_title h ("Distributed MNist Conv2d Adam 0.01 (2W)");
    set_xrange h 0. (x_range +. 10.);
    set_yrange h 0. (y_range losses 2.0);
    set_xlabel h "Time (s)";
    set_ylabel h "Loss";
    set_font_size h 8.;
    set_pen_size h 3.;
    let t' = Array.of_list time in
    let x = Owl.Mat.of_array t' 1 (Array.length t') in
    let l' = Array.of_list losses in
    let y = Owl.Mat.of_array l' 1 (Array.length l') in
    plot ~h x y;
    
    output h;; *)
    
    let open Printf in
    let file = "losstime.txt" in
    (* Write losses to file to print graph separately*)
    let oc = open_out file in 
    fprintf oc "[";
    List.iter (fprintf oc "%.6f,") losses;
    fprintf oc "]\n";     
    fprintf oc "[";
    List.iter (fprintf oc "%.6f,") time;
    fprintf oc "]\n";
    close_out oc

  let overwrite_file filename l = 
    let open Printf in
    if Sys.file_exists filename then
      let ic = open_in filename in
      try 
        let l' = input_line ic in 
        close_in ic;
        let oc = open_out filename in
        let total = l +. (float_of_string l') in 
        fprintf oc "%.6f" total;
        close_out oc  
      with e ->                      
        close_in_noerr ic;           
        raise e                      
    else
      let oc = open_out filename in 
      fprintf oc "%.6f" l;
      close_out oc


  let write_float_to_file filename l =
    let open Printf in
    let oc = open_out_gen [Open_creat; Open_text; Open_append] 0o640 filename in
    fprintf oc "%.6f," l;
    close_out oc  

  (* TODO: Refactor brute-force approach. *)
  let test_network task =
    Owl_log.info "Running test";
    let model = M.load "model" in
    let open Owl_dense in
    let imgs, labels = unpack_arr task.test_x, unpack_arr task.test_y in

    let s1 = [ [0;1999] ] in
    let s2 = [ [2000;3999] ] in
    let s3 = [ [4000;5999] ] in
    let s4 = [ [6000;7999] ] in
    let s5 = [ [8000;9999] ] in
    let imgs1 = Ndarray.S.get_slice s1 imgs in
    let imgs2 = Ndarray.S.get_slice s2 imgs in
    let imgs3 = Ndarray.S.get_slice s3 imgs in
    let imgs4 = Ndarray.S.get_slice s4 imgs in
    let imgs5 = Ndarray.S.get_slice s5 imgs in

    (* Assume all slices same size *)
    let m = Ndarray.S.nth_dim imgs1 0 in

    let mat2num x s = Matrix.S.of_array (
        x |> Matrix.Generic.max_rows
          |> Array.map (fun (_,_,num) -> float_of_int num)
      ) 1 s
    in
    let pred1 = mat2num (M.model model imgs1) m in
    Owl_log.info "Calculating.. 20";
    let pred2 = mat2num (M.model model imgs2) m in
    Owl_log.info "Calculating.. 40";
    let pred3 = mat2num (M.model model imgs3) m in
    Owl_log.info "Calculating.. 60";
    let pred4 = mat2num (M.model model imgs4) m in
    Owl_log.info "Calculating.. 80";
    let pred5 = mat2num (M.model model imgs5) m in

    let pred = Matrix.S.concat_horizontal pred1 pred2 in
    let pred = Matrix.S.concat_horizontal pred pred3 in
    let pred = Matrix.S.concat_horizontal pred pred4 in
    let pred = Matrix.S.concat_horizontal pred pred5 in

    let fact = mat2num labels (m * 5) in
    let accu = Matrix.S.(elt_equal pred fact |> sum') in
    let res = (accu /. (float_of_int (m * 5))) in
    Owl_log.info "Accuracy on test set: %f" res;
    write_float_to_file "result.txt" res

  let validate_model task i = 
    Owl_log.debug "Validation iteration: %i" i;
    let x = task.val_x in
    let y = task.val_y in
    let model = M.copy task.model in
    let xt, yt = (Batch.run task.server_params.batch) x y i in
    let yt', _ = (M.forward model) xt  in
    let loss = (Loss.run task.server_params.loss) yt yt' in
    (* take the mean of the loss *)
    let loss = Maths.(loss / (F (Mat.row_num yt |> float_of_int))) in
    Owl_log.debug "Validation Loss = %.6f." (unpack_flt loss);
    unpack_flt loss 

  (* retrieve local model at parameter server, init if none *)
  let local_model task =
    try E.get task.sid |> fst
    with Not_found -> (
      Owl_log.debug "set up first model";
      M.init task.model;
      E.set task.sid task.model;
      E.get task.sid |> fst;
    )

  (* retrieve the number of update iterations at parameter server for AdaDelay *)
  let local_iteration task =
    let k = (string_of_int task.sid ^ "iter") in
    try E.get k |> fst
    with Not_found -> (
      E.set k 0;
      E.get k |> fst;
    )

  (* retrieve the total gradient for Adaptive Revision *)
  let total_gradient task =
    let k = (string_of_int task.sid ^ "total_grad") in
    try E.get k |> fst
    with Not_found -> (
      E.set k (Owl_utils.aarr_map (fun _ -> F 0.) (M.mkpar task.model));
      E.get k |> fst;
    )

  let calc_implicit_momentum num_of_workers =
    unpack_flt Maths.(F 1. - (F 1. / F (float_of_int num_of_workers)))


  (* base batch_size for PASP *)
  let base_bs task =
    let params = task.server_params in
    let k = (string_of_int task.sid ^ "base_bs") in
    try E.get k |> fst
    with Not_found -> (
      let bs = match params.batch with
                            | Sample b           -> b 
                            | Mini b             -> b
                            | Stochastic         -> 1
                            | Full               -> 0
      in
      E.set k bs;
      E.get k |> fst;
    )

  (* current batch_size for PASP *)
  let current_bs task =
    let params = task.server_params in
    let k = (string_of_int task.sid ^ "current_bs") in
    try E.get k |> fst
    with Not_found -> (
      E.set k params.batch;
      E.get k |> fst;
    )


  (* for learning_rate for PASP *)
  let base_lr task =
    let params = task.server_params in
    let k = (string_of_int task.sid ^ "base_lr") in
    try E.get k |> fst
    with Not_found -> (
      let a = match params.learning_rate with
                            | Adagrad a          -> a
                            | Const a            -> a
                            | AdaptiveRev a      -> a
                            | AdaDelay a         -> a
                            | DelayComp (a, v, m)-> a
                            | _                  -> 0.
      in
      E.set k a;
      E.get k |> fst;
    )

  let base_workers task =
    let k = (string_of_int task.sid ^ "base_workers") in
    try E.get k |> fst
    with Not_found -> (
      E.set k (E.progressive_num ());
      E.get k |> fst;
    )

  (* retrieve total_momentum for PASP *)
  let total_momentum task = 
    let params = task.server_params in
    let k = (string_of_int task.sid ^ "total_momentum") in
    try E.get k |> fst
    with Not_found -> (
      let implicit_momentum = E.progressive_num () |> calc_implicit_momentum in 
      let explicit_momentum = match params.momentum with
      | Standard m -> m
      | Nesterov m -> m
      | None       -> 0.0
      in
      E.set k (explicit_momentum +. implicit_momentum);
      Owl_log.debug "explicit_momentum: %f" explicit_momentum;
      Owl_log.debug "implicit_momentum: %f" implicit_momentum;
      E.get k |> fst;
    )

  (* Decay duration for worker switch *)
  let decay_duration task = 
    let k = (string_of_int task.sid ^ "decay_duration") in
    try E.get k |> fst
    with Not_found -> (
      E.set k 0;
      E.get k |> fst;
    )

  let exit_condition task_id =
    try E.get (string_of_int task_id ^ "finish") |> fst
    with Not_found -> false

  let schedule task workers =
    let params = task.server_params in
    (* get model, if none then init locally *)
    let model = local_model task in
    let batch_no = task.schedule_no in
    let cb = current_bs task in
    (* If AdaptiveRevision, record total gradient for worker before schedule.
       If AdaDelay, record current iteration *)
    let tasks = List.mapi (fun i x ->
      let _ = match params.learning_rate with 
      | AdaptiveRev _   -> let total_gs = total_gradient task in
                           E.set (x ^ "gradient") total_gs
      | AdaDelay _      -> let iter = local_iteration task in
                           E.set (x ^ "iter") iter
      | DelayComp _     -> E.set (x ^ "model") (M.mkpar model)
      | _               -> () in
      E.set (x ^ "time") (Unix.gettimeofday ());
      (x, [(task.sid, (model, batch_no + i, cb))])
    ) workers
    in
    task.schedule_no <- batch_no + (List.length tasks);
    tasks

  let pull task address vars =
    let n = E.worker_num () |> float_of_int in
    assert (n >= 1.); (* at least one worker *)
    (* there should be only one item in list *)
    List.map (fun (k, v) ->
      let gradient, loss = v in
      let schedule_time = E.get (address ^ "time") |> fst in
      let response_time = (Unix.gettimeofday () -. schedule_time) in
      write_float_to_file "respond.txt" response_time;
      let params = task.server_params in
      let xs = task.train_size in
      let model = local_model task in
      (* Hotfix: initialize total_momentum. Might need create pre-start function *)
      (* let _ = total_momentum task in  *)
      let _ = base_workers task in
      let _ = base_bs task in
      let _ = base_lr task in
      (* Calculate delay for revised learning rate *)
      let delay = match params.learning_rate with
      | AdaDelay _ -> let iter = local_iteration task in
                      let prev_iter = E.get (address ^ "iter") |> fst in
                      E.set (string_of_int task.sid ^ "iter") (iter + 1);
                      iter - prev_iter
      | _          -> 0
      in
      (* Calculate gradient for revision step *)
      let gradient_back = match params.learning_rate with 
      | AdaptiveRev _ -> let gradient_old = E.get (address ^ "gradient") |> fst in
                         let total_gs = total_gradient task in
                         Owl_utils.aarr_map2 (fun w u -> Maths.(w - u)) total_gs gradient_old
      | DelayComp _   -> let model_old = E.get (address ^ "model") |> fst in
                         Owl_utils.aarr_map2 (fun w u -> Maths.(w - u)) (M.mkpar model) model_old
      | _             -> Owl_utils.aarr_map (fun _ -> F 0.) gradient
      in
      let gradients = gradient, gradient_back in
      let state = match task.state with
        | Some state -> M.update_network ~state ~params ~init_model:false model gradients delay loss xs
        | None       -> M.update_network ~params ~init_model:false model gradients delay loss xs
      in
      
      task.model <- model;
      E.set task.sid task.model;
      task.state <- Some state;        

      (* Update total gradient in AdaptiveRevision *)
      let _ = match params.learning_rate with 
      | AdaptiveRev _ -> let total_gs = total_gradient task in
                         E.set (string_of_int task.sid ^ "total_grad") 
                         (Owl_utils.aarr_map2 (fun w u -> Maths.(w + u)) total_gs gradient)
      | _             -> () 
      in

      let t = Unix.gettimeofday () -. task.start_at in
      let loss' = unpack_flt loss in

      (* Writing loss and time to file for further analysis *)
      write_float_to_file "loss.txt" loss';
      write_float_to_file "time.txt" t;

      (* For plotting in real time. Bug in Ubuntu 16.04 Server. *)
      (* task.loss <- loss' :: task.loss;
      task.time <- t :: task.time;
      plot_loss_time task.loss task.time; *) 

      (* Calculate Validation loss every epoch *)
      let _ = match Checkpoint.(state.current_batch mod (state.batches_per_epoch) = 0) with
        | false ->  ()
        | true  ->  let vl = validate_model task (Checkpoint.(state.current_batch / (state.batches_per_epoch)) - 1) in
                    write_float_to_file "val_loss.txt" vl;
                    match task.lowest_val_loss <> 0. && vl >= task.lowest_val_loss with
                      | true  ->  task.patience <- task.patience + 1
                      | false ->  M.save task.model "model";
                                  task.lowest_val_loss <- vl;
                                  task.patience <- 0
      in

      (* Determine if training ends *)
      let _ = match task.patience >= 100 with 
      | false -> ()
      | true  -> Owl_log.info "Early stopping..";
                 Checkpoint.(state.stop <- true)
      in
      E.set (string_of_int task.sid ^ "finish") Checkpoint.(state.stop); 

(*       let current_progression = E.progressive_num () in
      (* Add/Remove workers for PASP barrier every 125 iterations *)
      let workers_changed = match Checkpoint.(state.current_batch mod 125 = 0) with
        | false -> false
        (* Capricious mode *)
        | true  -> let b  = Owl_stats.uniform_int_rvs ~a:0 ~b:1 in
                   let cw = Owl_stats.uniform_int_rvs ~a:1 ~b:8 in
                   match b with
                   | 1 -> Owl_log.debug "%i workers attempting to join." cw;
                          E.add_workers cw
                   | _ -> Owl_log.debug "%i workers attempting to leave." cw;
                          E.remove_workers cw
        
        (* Progressive mode *)
        (* | true  -> E.add_workers current_progression *)
      in
 *)
      (* Detect if workers changed *)
(*       let _ = match workers_changed with
        | false ->  ()
        | true  ->  (* Increase batch size *)
                    let bs = base_bs task |> float_of_int in
                    let lr = base_lr task in
                    Owl_log.debug "Base BS: %f" bs;
                    let w = base_workers task in
                    let w' = E.progressive_num () in
                    let d = (w' - w) in
                    Owl_log.debug "Worker count changed to %i" w';
                    let d = float_of_int d in
                    let nlr = lr *. (exp (-0.2 *. d)) in
                    let nbs = bs *. (lr /. nlr) |> int_of_float in
                    Owl_log.debug "New Batch Size %i" nbs;
                    (* Check if new batch size affects training *)
                    let batches_per_epoch = match params.batch with
                          | Full       -> 1
                          | Mini _     -> xs / nbs
                          | Sample _   -> xs / nbs
                          | Stochastic -> xs
                          in
                    let batches = (float_of_int batches_per_epoch) *. params.epochs |> int_of_float in
                    Owl_log.debug "Iterations: %i" batches;
                    match Checkpoint.(state.current_batch) >= batches with
                    | true  -> Owl_log.debug "Batch size too big. Removing recently added workers..";
                               let _ = E.remove_workers (w' - current_progression) in
                               ()
                    | false ->
                      let _ = match params.batch with
                      | Sample _    -> params.batch <- Sample nbs
                      | Mini _      -> params.batch <- Mini nbs
                      | Stochastic  -> params.batch <- Mini nbs
                      | Full        -> ()
                      in
                      E.set (string_of_int task.sid ^ "current_bs") params.batch;

                      (* Decay learning rate *)
                  (*  let lr = base_lr task in
                      let w  = base_workers task in
                      let w' = E.progressive_num () in
                      let d  = (w' - w) in
                      E.set (string_of_int task.sid ^ "decay_duration") (d * 20);
                      Owl_log.debug "Worker count changed to %i" w';
                      (*Owl_log.debug "Set decay duration to %i batches" (d * 20);*)
                      let d = float_of_int d in
                      let nlr = lr *. (exp (-0.1 *. d)) in*)


                      Owl_log.debug "New Learning Rate: %f" nlr;
                      match params.learning_rate with
                      | Adagrad _          -> params.learning_rate <- Adagrad nlr
                      | Const _            -> params.learning_rate <- Const nlr
                      | AdaptiveRev _      -> params.learning_rate <- AdaptiveRev nlr
                      | AdaDelay _         -> params.learning_rate <- AdaDelay nlr
                      | DelayComp (_, v, m)-> params.learning_rate <- DelayComp (nlr, v, m)
                      | _                  -> ()
 *)


                      (* Change momentum. Doesn't work with adaptive learning algos. *)
                      (* let w = E.progressive_num () in
                      let tm = total_momentum task in
                      let im = calc_implicit_momentum w in
                      let em = (tm -. im) in
                      Owl_log.debug "Worker count changed to %i" w;
                      Owl_log.debug "Total momentum: %f. New implicit momentum: %f." tm im;
                      Owl_log.debug "Setting new explicit momentum: %f." em;
                      match params.momentum with
                        | Standard _ -> params.momentum <- Momentum.Standard em
                        | Nesterov _ -> params.momentum <- Momentum.Nesterov em
                        | None -> params.momentum <- Momentum.Standard em *)
      (* in *)
      (* Detect if decay expired *)
(*    let decay = decay_duration task in
      let _ = match (decay <> 0 
                    && Checkpoint.(state.current_batch mod (state.batches_per_epoch * 1 + decay) = 0)) with
        | false -> ()
        | true -> let lr = base_lr task in
                  Owl_log.debug "Decay expired..";
                  E.set (string_of_int task.sid ^ "decay_duration") 0;
                  match params.learning_rate with
                  | Adagrad _          -> Owl_log.debug "New Learning Rate: %f" lr;
                                          params.learning_rate <- Adagrad lr
                  | Const _            -> Owl_log.debug "New Learning Rate: %f" lr;
                                          params.learning_rate <- Const lr
                  | AdaptiveRev _      -> Owl_log.debug "New Learning Rate: %f" lr;
                                          params.learning_rate <- AdaptiveRev lr
                  | AdaDelay _         -> Owl_log.debug "New Learning Rate: %f" lr;
                                          params.learning_rate <- AdaDelay lr
                  | DelayComp (_, v, m)-> Owl_log.debug "New Learning Rate: %f" lr;
                                          params.learning_rate <- DelayComp (lr, v, m)
                  | _                  -> ()
      in *)
          
      let _ = match Checkpoint.(state.stop) with
        | true  -> test_network task
        | false -> ()
      in

      (k, model)
    ) vars

  let push task id vars =
    (* there should be only one item in list *)
    List.map (fun (k, v) ->
      let model, t, bs = v in
      let start_t = Unix.gettimeofday () in
      task.client_params.batch <- bs;
      (* start local training *)
      let params = task.client_params in
      let x = task.train_x in
      let y = task.train_y in
      let grad, loss = M.calculate_gradient ~params ~init_model:false model x y t in
      let result = (grad, loss) in
      write_float_to_file "computation.txt" (Unix.gettimeofday () -. start_t);
      (k, result)      
       ) vars 

  (* Stop scheduling if model finishes training *)
  let stop task context = exit_condition task.sid

  let train_generic ?params nn x y tx ty jid url =
    (* prepare params and make task *)
    let params = match params with
      | Some p -> p
      | None   -> Params.default ()
    in
    let sid = Owl_stats.uniform_int_rvs ~a:0 ~b:max_int in
    let cid = Owl_stats.uniform_int_rvs ~a:0 ~b:max_int in

    (* Split training and validation data to 80:20 *)
    let r = Array.init (Owl_dense_ndarray.S.nth_dim x 0) (fun i -> i) in
    let r = Owl_stats.shuffle r in
    
    (* Validation data *)
    let v_rows = Array.sub r 0 2000 in 
    let vx = Arr (Owl_dense_ndarray.S.get_fancy [L (Array.to_list v_rows)] x) in
    let vy = Arr (Owl_dense_ndarray.S.rows y v_rows) in

    (* Training data *)
    let t_rows = Array.sub r 2000 8000 in
    let x = Arr (Owl_dense_ndarray.S.get_fancy [L (Array.to_list t_rows)] x) in
    let y = Arr (Owl_dense_ndarray.S.rows y t_rows) in

    let server_task = make_server_task sid params nn x vx vy tx ty in
    let client_task = make_client_task cid params x y in 
    (* register sched/push/pull/stop/barrier *)
    E.register_schedule (schedule server_task);
    E.register_pull (pull server_task);
    E.register_push (push client_task);
    E.register_stop (stop server_task);
    E.start ~barrier:E.ASP jid url


  let train ?params nn x y tx ty jid url = train_generic ?params nn x y 
                                            (Arr tx) (Arr ty) jid url


end
