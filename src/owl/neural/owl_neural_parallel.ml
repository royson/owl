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

  val register_pull : (('a * (t array array * t)) list -> ('a * 'd) list) -> unit

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

  (* val forward : network -> t -> t * t array array *)

  (* val train_generic : ?state:Checkpoint.state -> ?params:Params.typ -> ?init_model:bool -> network -> t -> t -> Checkpoint.state *)

  val calculate_gradient : ?params:Params.typ -> ?init_model:bool -> network -> t -> t -> t array array * t

  val update_network : ?state:Checkpoint.state -> ?params:Params.typ -> ?init_model:bool -> network -> t array array -> 
                        t -> t -> Checkpoint.state

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

  let exit_condition task_id =
    try E.get (string_of_int task_id ^ "finish") |> fst
  with Not_found -> false



  let schedule task workers =
    (* get model, if none then init locally *)
    let model = local_model task in
    let tasks = List.map (fun x ->
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

  let pull task vars =
    let n = E.worker_num () |> float_of_int in
    assert (n >= 1.); (* at least one worker *)
    (* Owl_log.warn "PULL!"; *)
    (* there should be only one item in list *)
    List.map (fun (k, v) ->
      let gradient, loss = v in
      let params = task.params in
      let x = task.data_x in
      let model = local_model task in
      let state = match task.state with
        | Some state -> M.(update_network ~state ~params ~init_model:false model gradient loss x)
        | None       -> M.(update_network ~params ~init_model:false model gradient loss x)
      in
      
      task.state <- Some state; 
      task.model <- model;
      (* Model is saved in on return *)
      (* E.set task.id task.model; *)
      E.set (string_of_int task.id ^ "finish") Checkpoint.(state.stop);
      
      (* Calculate loss for Model *)
      (* let params = task.params in
      let x = task.data_x in
      let y = task.data_y in
      let loss = calc_loss (M.copy task.model) params x y in
       *)
      let t = Unix.gettimeofday () -. task.start_at in
      task.loss <- (unpack_flt loss) :: task.loss;
      task.time <- t :: task.time;
      (* plot_loss task.loss; *) (* Plot Loss * Update *)
      plot_loss_time task.loss task.time; (* Plot Loss * Time *)
      (k, model)
    ) vars


  let push task id vars =
    (* there should be only one item in list *)
    List.map (fun (k, model) ->
      (* start local training *)
      let params = task.params in
      let x = task.data_x in
      let y = task.data_y in
      let result = M.(calculate_gradient ~params ~init_model:false model x y) in
      (k, result)      
       ) vars 

  (* Stop scheduling if model finishes training *)
  let stop task context = exit_condition task.id


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
