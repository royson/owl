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

  val register_pull : (string -> ('a * ('b * 'd)) list -> ('a * 'c) list) -> unit

  val register_push : ('a -> ('b * 'c) list -> (('b * ('c * 'e)) list * 'd)) -> unit

  val register_stop : (param_context ref -> bool) -> unit

end


(* module signature of neural network model *)

module type ModelSig = sig

  type network

  val mkpar : network -> t array array

  val init : network -> unit

  val update : network -> t array array -> unit

  val copy : network -> network

  val forward: network -> t -> t * t array array

  val train_generic : ?state:Checkpoint.state -> ?params:Params.typ -> ?init_model:bool -> network -> t -> t -> Checkpoint.state

end


(* implementation of parallel neural network training *)

module Make (M : ModelSig) (E : EngineSig) = struct

  type task = {
    mutable id        : int;
    mutable state     : Checkpoint.state option;
    mutable params    : Params.typ;
    mutable model     : M.network;
    mutable data_x    : t;
    mutable data_y    : t;
    mutable loss      : float list; (* For graph plot *)
    mutable start_at  : float;      (* Time training starts *)
    mutable time      : float list; (* For graph plot *)
  }


  let make_task id params model data_x data_y = {
    id;
    state = None;
    params;
    model;
    data_x;
    data_y;
    loss = [];
    start_at = Unix.gettimeofday ();
    time = [];
  }


  (* calculate \delta model = model0 - model1, save the result in model0 *)
  let delta_model model0 model1 =
    let par0 = M.mkpar model0 in
    let par1 = M.mkpar model1 in
    let delta = Owl_utils.aarr_map2 (fun a0 a1 -> Maths.(a0 - a1)) par0 par1 in
    M.update model0 delta

  let write_float_to_file filename l =
    let open Printf in
    let file = filename in
    let oc = open_out_gen [Open_creat; Open_text; Open_append] 0o640 file in
    fprintf oc "%.6f," l;
    close_out oc

  (* Plot the loss function per update *)
  let plot_loss losses =
    let open Owl_plot in
    (* Might replace List with an Array with a length counter instead *)
    let losses = List.rev losses in
    
    (* Debug. Print losses. *)
    (* List.map (Owl_log.info "Loss: %.6f") losses; *)

    (* Bug: Currently crashes on Ubuntu 16.04. Writing to file instead.*)
    (* let h = create ("Distributed MNist MLP Adam 0.01 2W.png") in
    let f x = (int_of_float (x -. 1.)) |> (List.nth losses) in
    let x_range = float_of_int ((List.length losses)) in
    let y_range l = List.fold_left Pervasives.max (List.hd l) l |> Pervasives.(+.) in
    Owl_log.debug "Plotting loss function.. ";
    set_foreground_color h 0 0 0;
    set_background_color h 255 255 255;
    set_title h ("Distributed MNist MLP Adam 0.01 (2W)");
    set_xrange h 0. (x_range +. 10.);
    set_yrange h 0. (y_range losses 2.0);
    set_xlabel h "Updates";
    set_ylabel h "Loss";
    set_font_size h 8.;
    set_pen_size h 3.;

    plot_fun ~h f 1. x_range;
    
    output h;; *)
    
    let open Printf in
    let file = "loss.txt" in
    (* Write losses to file to print graph separately*)
    let oc = open_out file in 
    fprintf oc "[";
    List.iter (fprintf oc "%.6f;") losses;
    fprintf oc "]";      
    close_out oc                

  (* Plot the loss function per time *)
  let plot_loss_time losses time =
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
    List.iter (fprintf oc "%.6f;") losses;
    fprintf oc "]\n";     
    fprintf oc "[";
    List.iter (fprintf oc "%.6f;") time;
    fprintf oc "]";
    close_out oc

                   

  (* retrieve local model at parameter server, init if none *)
  let local_model task =
    try E.get task.id |> fst
    with Not_found -> (
      Owl_log.warn "set up first model";
      M.init task.model;
      E.set task.id task.model;
      E.get task.id |> fst;
    )


  let schedule task workers =
    (* get model, if none then init locally *)
    let model = local_model task in
    let tasks = List.map (fun x ->
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
    (* Owl_log.warn "PULL!"; *)
    (* there should be only one item in list *)
    List.map (fun (k, v) ->
      let model1, loss = v in
      let schedule_time = E.get (address ^ "time") |> fst in
      let response_time = (Unix.gettimeofday () -. schedule_time) in
      write_float_to_file "respond.txt" response_time;
      let model0 = local_model task in
      let par0 = M.mkpar model0 in
      let par1 = M.mkpar model1 in
      Owl_utils.aarr_map2 (fun a0 a1 ->
        Maths.(a0 + a1)
      ) par0 par1
      |> M.update model0;
      task.model <- model0;
      E.set task.id task.model;
      (* Calculate loss for Model *)
(*       let params = task.params in
      let x = task.data_x in
      let y = task.data_y in
      let loss = calc_loss (M.copy task.model) params x y in *)
      let t = Unix.gettimeofday () -. task.start_at in
      task.loss <- loss :: task.loss;
      task.time <- t :: task.time;
      (* plot_loss task.loss; *) (* Plot Loss * Update *)
      (* plot_loss_time task.loss task.time; *) (* Plot Loss * Time *)
      write_float_to_file "loss.txt" loss;
      write_float_to_file "time.txt" t;
      (k, model0)
    ) vars


  let push task id vars =
    (* there should be only one item in list *)
    let updates = List.map (fun (k, model) ->
      let start_t = Unix.gettimeofday () in
      task.model <- M.copy model;
      (* start local training *)
      let params = task.params in
      let x = task.data_x in
      let y = task.data_y in
      let state = match task.state with
        | Some state -> if state.batches <> (state.current_batch) then
                          M.(train_generic ~state ~params ~init_model:false model x y)
                        else
                          state
        | None       -> M.(train_generic ~params ~init_model:false model x y)
      in
      let loss = state.loss.(state.current_batch - 1) |> unpack_flt in
      Checkpoint.(state.stop <- false);
(*       Owl_log.warn "PUSH!";
      Owl_log.warn "BATCHES: %i" state.batches; 
      Owl_log.warn "CURRENT BATCH: %i" state.current_batch;
      Owl_log.warn "ID: %i" id;
      Owl_log.warn "Task ID: %i" task.id; *)
      task.state <- Some state;
      
      (* only send out delta model *)
      delta_model model task.model;
      write_float_to_file "computation.txt" (Unix.gettimeofday () -. start_t);
      
      (k, (M.copy model, loss))      
       ) vars in
      match task.state with
        | Some state -> 
            if state.batches = (state.current_batch) then
              (updates, true)
            else
              (updates, false)
        | None -> (updates, false)
        (* | None -> failwith "Task not executed" *)

  (* FIXME: currently check is done in actor *)
  let stop task _context = false
    (* !_context.finish = StrMap.cardinal !_context.workers *)


  let train_generic ?params nn x y jid url =
    (* prepare params and make task *)
    let params = match params with
      | Some p -> p
      | None   -> Params.default ()
    in
    let id = Owl_stats.uniform_int_rvs ~a:0 ~b:max_int in
    let task = make_task id params nn x y in
    (* register sched/push/pull/stop/barrier *)
    E.register_schedule (schedule task);
    E.register_pull (pull task);
    E.register_push (push task);
    E.register_stop (stop task);
    E.start ~barrier:E.ASP jid url


  let train ?params nn x y jid url = train_generic ?params nn (Arr x) (Arr y) jid url


end
