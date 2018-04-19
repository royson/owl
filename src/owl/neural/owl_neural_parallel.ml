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
  type barrier = ASP | BSP | SSP | PSP

  (* functions of parameter server engine *)

  val get : 'a -> 'b * int

  val set : 'a -> 'b -> unit

  val worker_num : unit -> int

  val start : ?barrier:barrier -> string -> string -> unit

  val register_barrier : (param_context ref -> int * (string list)) -> unit

  val register_schedule : ('a list -> ('a * ('b * 'c) list) list) -> unit

  val register_pull : (string -> ('a * ('b * 'c)) list -> ('a * 'd) list) -> unit

  val register_push : ('a -> ('b * 'c) list -> ('b * ('d * 'e)) list) -> unit

  val register_stop : (param_context ref -> bool) -> unit

end


(* module signature of neural network model *)

module type ModelSig = sig

  type network

  val mkpar : network -> t array array

  val init : network -> unit

  val update : network -> t array array -> unit

  val copy : network -> network

  val model : network -> arr -> arr

  (* val forward : network -> t -> t * t array array *)

  (* val train_generic : ?state:Checkpoint.state -> ?params:Params.typ -> ?init_model:bool -> network -> t -> t -> Checkpoint.state *)

  val calculate_gradient : ?params:Params.typ -> ?init_model:bool -> network -> t -> t -> t array array * t

  val update_network : ?state:Checkpoint.state -> ?params:Params.typ -> ?init_model:bool -> network 
                      -> (t array array * t array array) -> int -> t -> t -> Checkpoint.state

end


(* implementation of parallel neural network training *)

module Make (M : ModelSig) (E : EngineSig) = struct

  type task = {
    mutable id          : int;
    mutable state       : Checkpoint.state option;
    mutable params      : Params.typ;
    mutable model       : M.network;
    mutable data_x      : t;          
    mutable data_y      : t;
    mutable test_x      : t;          
    mutable test_y      : t;
    mutable loss        : float list; (* Losses received *)
    mutable start_at    : float;      (* Time training starts *)
    mutable time        : float list; (* Total time executed by task *)
    mutable total_gs    : t array array; (* Total gradient for AdaptiveRevision *) 
  }


  let make_task id params model data_x data_y test_x test_y = {
    id;
    state = None;
    params;
    model;
    data_x;
    data_y;
    test_x;
    test_y;
    loss = [];
    start_at = Unix.gettimeofday ();
    time = [];
    total_gs = Owl_utils.aarr_map (fun _ -> F 0.) (M.mkpar model);
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

  let write_float_to_file filename l =
    let open Printf in
    let file = filename in
    let oc = open_out_gen [Open_creat; Open_text; Open_append] 0o640 file in
    fprintf oc "%.6f," l;
    close_out oc  
  
  (* TODO: Shape is hard-coded now *)
  let test task =
    let open Owl_dense in
    let imgs, labels = unpack_arr task.test_x, unpack_arr task.test_y in
    let m = Matrix.S.row_num labels in
    (* let imgs = Ndarray.S.reshape imgs [|m;32;32;3|] in *)
    let imgs = Ndarray.S.reshape imgs [|m;28;28;1|] in

    let mat2num x = Matrix.S.of_array (
        x |> Matrix.Generic.max_rows
          |> Array.map (fun (_,_,num) -> float_of_int num)
      ) 1 m
    in
    
    let pred = mat2num (M.model task.model imgs) in
    let fact = mat2num labels in
    let accu = Matrix.S.(elt_equal pred fact |> sum') in
    let res = (accu /. (float_of_int m)) in
    Owl_log.info "Accuracy on test set: %f" res;
    write_float_to_file "result.txt" res
            

  (* retrieve local model at parameter server, init if none *)
  let local_model task =
    try E.get task.id |> fst
    with Not_found -> (
      Owl_log.warn "set up first model";
      M.init task.model;
      E.set task.id task.model;
      E.get task.id |> fst;
    )

  (* retrieve the number of update iterations at parameter server *)
  let local_iteration task =
    let k = (string_of_int task.id ^ "iter") in
    try E.get k |> fst
    with Not_found -> (
      E.set k 0;
      E.get k |> fst;
    )


  let exit_condition task_id =
    try E.get (string_of_int task_id ^ "finish") |> fst
    with Not_found -> false

  let schedule task workers =
    let params = task.params in
    (* get model, if none then init locally *)
    let model = local_model task in
    (* If AdaptiveRevision, record total gradient for worker before schedule.
       If AdaDelay, record current iteration *)
    let tasks = List.map (fun x ->
      let _ = match params.learning_rate with 
      | AdaptiveRev _ -> E.set (x ^ "gradient") task.total_gs
      | AdaDelay _    -> let iter = local_iteration task in
                         E.set (x ^ "iter") iter;
      | _             -> () in
      E.set (x ^ "time") (Unix.gettimeofday ());
      (x, [(task.id, model)])
    ) workers
    in
    tasks


(*   let calc_loss model (params:Params.typ) x y =
    let xt, yt = (Batch.run params.batch) x y 0 in
    let yt', _ = (M.forward model) xt  in
    let loss = (Loss.run params.loss) yt yt' in
    (* take the mean of the loss *)
    let loss = Maths.(loss / (F (Mat.row_num yt |> float_of_int))) in
    Owl_log.warn "Current Loss = %.6f." (unpack_flt loss);
    unpack_flt loss 
 *)

  let pull task address vars =
    let n = E.worker_num () |> float_of_int in
    assert (n >= 1.); (* at least one worker *)
    (* there should be only one item in list *)
    List.map (fun (k, v) ->
      let gradient, loss = v in
      let schedule_time = E.get (address ^ "time") |> fst in
      let response_time = (Unix.gettimeofday () -. schedule_time) in
      write_float_to_file "respond.txt" response_time;
      let params = task.params in
      let x = task.data_x in
      let model = local_model task in

      (* Calculate delay for revised learning rate *)
      let delay = match params.learning_rate with
      | AdaDelay _ -> let iter = local_iteration task in
                      let prev_iter = E.get (address ^ "iter") |> fst in
                      let d = iter - prev_iter in
                      E.set (string_of_int task.id ^ "iter") (iter + 1);
                      d
      | _          -> 0
      in
      (* Calculate gradient for revision step *)
      let gradient_back = match params.learning_rate with 
      | AdaptiveRev _ -> let gradient_old = E.get (address ^ "gradient") |> fst in
                              Owl_utils.aarr_map2 (fun w u -> Maths.(w - u)) task.total_gs gradient_old
      | _             -> Owl_utils.aarr_map (fun _ -> F 0.) gradient
      in
      let gradients = gradient, gradient_back in
      let state = match task.state with
        | Some state -> M.update_network ~state ~params ~init_model:false model gradients delay loss x
        | None       -> M.update_network ~params ~init_model:false model gradients delay loss x
      in
      
      task.model <- model;
      E.set task.id task.model;
      task.state <- Some state; 

      (* Update total gradient in AdaptiveRevision *)
      let _ = match params.learning_rate with 
      | AdaptiveRev _ -> task.total_gs <- Owl_utils.aarr_map2 (fun w u -> Maths.(w + u)) task.total_gs gradient
      | _             -> () 
      in
      
      (* Calculate loss for Model *)
      (* let params = task.params in
      let x = task.data_x in
      let y = task.data_y in
      let loss = calc_loss (M.copy task.model) params x y in
       *)
      let t = Unix.gettimeofday () -. task.start_at in
      let loss' = unpack_flt loss in
      task.loss <- loss' :: task.loss;
      task.time <- t :: task.time;
      write_float_to_file "loss.txt" loss';
      write_float_to_file "time.txt" t;
      (* plot_loss_time task.loss task.time; *) (* Plot Loss * Time *)

      if Checkpoint.(state.stop) then
        E.set (string_of_int task.id ^ "finish") true;
        test task;
      (k, model)
    ) vars


  let push task id vars =
    (* there should be only one item in list *)
    List.map (fun (k, model) ->
      let start_t = Unix.gettimeofday () in
      (* start local training *)
      let params = task.params in
      let x = task.data_x in
      let y = task.data_y in
      let grad, loss = M.calculate_gradient ~params ~init_model:false model x y in
      let result = (grad, loss) in
      write_float_to_file "computation.txt" (Unix.gettimeofday () -. start_t);
      (k, result)      
       ) vars 

  (* Stop scheduling if model finishes training *)
  let stop task context = exit_condition task.id

  let train_generic ?params nn x y tx ty jid url =
    (* prepare params and make task *)
    let params = match params with
      | Some p -> p
      | None   -> Params.default ()
    in
    let id = Owl_stats.uniform_int_rvs ~a:0 ~b:max_int in

    let task = make_task id params nn x y tx ty in
    (* register sched/push/pull/stop/barrier *)
    E.register_schedule (schedule task);
    E.register_pull (pull task);
    E.register_push (push task);
    E.register_stop (stop task);
    E.start ~barrier:E.ASP jid url


  let train ?params nn x y tx ty jid url = train_generic ?params nn (Arr x) (Arr y) 
                                            (Arr tx) (Arr ty) jid url


end
