#!/usr/bin/env owl
(* This example shows how to use checkpoint in a stateful optimisation. *)

open Owl
open Neural.S
open Neural.S.Graph
open Algodiff.S


let make_network input_shape =
  input input_shape
(*   |> lambda (fun x -> Maths.(x / F 256.))
  |> conv2d [|5;5;1;32|] [|1;1|] ~act_typ:Activation.Relu
  |> max_pool2d [|2;2|] [|2;2|]
  |> dropout 0.1
  |> fully_connected 1024 ~act_typ:Activation.Relu
  |> linear 10 ~act_typ:Activation.Softmax
  |> get_network *)

(*     |> conv2d [|3;3;1;32|] [|1;1|] ~act_typ:Activation.Relu
    |> conv2d [|3;3;32;64|] [|1;1|] ~act_typ:Activation.Relu
    |> max_pool2d [|2;2|] [|2;2|]
    |> dropout 0.5
    |> flatten
    |> linear 128 ~act_typ:Activation.Relu
    |> normalisation
    |> dropout 0.5
    |> linear 10 ~act_typ:Activation.Softmax
    |> get_network *)
    (* MLP *)
  |> flatten 
  |> linear 256 ~act_typ:Activation.Tanh
  |> linear 128 ~act_typ:Activation.Relu
  |> linear 10 ~act_typ:Activation.Softmax
  |> get_network


let train () =
  let x, _, y = Dataset.load_mnist_train_data_arr () in
  let network = make_network [|28;28;1|] in

  (* define checkpoint function *)
  let chkpt state =
    let open Checkpoint in
    if state.current_batch mod (state.batches - 1) = 0 then (
(*       Owl_log.info "Plotting loss function..";   
      let z = Array.map unpack_flt state.loss in
      let c = Array.sub z 0 state.batches in 
      (* Array.map (Owl_log.info "%.6f") c; *)

      let h = Plot.create "seq_mnist_conv2d_Adam_0.01.png" in
      let f x = c.(Maths.(pack_flt x - F 1.) |> unpack_flt |> int_of_float) in
      let x_range = Maths.(F (float_of_int (Array.length c)) - F 1.) in
      let y_range arr = Array.fold_left Pervasives.max arr.(0) arr |> Pervasives.(+.) in
      Plot.set_foreground_color h 0 0 0;
      Plot.set_background_color h 255 255 255;
      Plot.set_title h "Sequential Adam 0.01";
      Plot.set_xrange h 0. (unpack_flt Maths.(x_range + F 10.));
      Plot.set_yrange h 0. (y_range c 2.0);
      Plot.set_xlabel h "Batch";
      Plot.set_ylabel h "Loss";
      Plot.set_font_size h 8.;
      Plot.set_pen_size h 3.;

      Plot.plot_fun ~h f 1. (unpack_flt x_range);

      Plot.output h; *)

      state.stop <- true;
    )
  in

  (* plug in chkpt into params *)
  let params = Params.config
    ~batch:(Batch.Sample 100) ~learning_rate:(Learning_Rate.Adam (0.01, 0.9, 0.999))
    ~checkpoint:(Checkpoint.Custom chkpt) ~stopping:(Stopping.Const 1e-6) 0.1
  in
  (* keep restarting the optimisation until it finishes *)
  let state = Graph.train ~params network x y in
  while Checkpoint.(state.current_batch < state.batches) do
    Checkpoint.(state.stop <- false);
    Graph.train ~state ~params ~init_model:false network x y |> ignore
  done;

  network


let _ = train ()
