#!/usr/bin/env owl
(* This example shows how to use checkpoint in a stateful optimisation. *)

open Owl
open Neural.S
open Neural.S.Graph
open Algodiff.S
open Owl_optimise.S

let write_float_to_file filename l =
  let open Printf in
  let oc = open_out_gen [Open_creat; Open_text; Open_append] 0o640 filename in
  fprintf oc "%.6f," l;
  close_out oc  


let make_network input_shape =
  input input_shape
  |> normalisation ~decay:0.9
  |> conv2d [|3;3;3;32|] [|1;1|] ~act_typ:Activation.Relu
  |> conv2d [|3;3;32;32|] [|1;1|] ~act_typ:Activation.Relu ~padding:VALID
  |> max_pool2d [|2;2|] [|2;2|] ~padding:VALID
  |> dropout 0.1
  |> conv2d [|3;3;32;64|] [|1;1|] ~act_typ:Activation.Relu
  |> conv2d [|3;3;64;64|] [|1;1|] ~act_typ:Activation.Relu ~padding:VALID
  |> max_pool2d [|2;2|] [|2;2|] ~padding:VALID
  |> dropout 0.1
  |> fully_connected 512 ~act_typ:Activation.Relu
  |> linear 10 ~act_typ:Activation.(Softmax 1)
  |> get_network

let validate_model (params:Params.typ) model i vx vy = 
  Owl_log.info "Validation iteration: %i" i;
  let model = Graph.copy model in
  let xt, yt = (Batch.run params.batch) vx vy i in
  let yt', _ = (Graph.forward model) xt in
  let loss = (Loss.run params.loss) yt yt' in
  (* take the mean of the loss *)
  let loss = Maths.(loss / (F (Mat.row_num yt |> float_of_int))) in
  Owl_log.info "Validation Loss = %.6f." (unpack_flt loss);
  unpack_flt loss 

let test model =
  let imgs, _, labels = Dataset.load_cifar_test_data () in
  let model = Graph.copy model in
  let mat2num x s = Dense.Matrix.S.of_array (
      x |> Dense.Matrix.Generic.max_rows
        |> Array.map (fun (_,_,num) -> float_of_int num)
    ) 1 s
  in

  let s = [ 
  [ [0;499] ] ; [ [500;999] ] ; [ [1000;1499] ] ; [ [1500;1999] ]
  ; [ [2000;2499] ] ; [ [2500;2999] ] ; [ [3000;3499] ] ; [ [3500;3999] ] 
  ; [ [4000;4499] ] ; [ [4500;4999] ] ; [ [5000;5499] ] ; [ [5500;5999] ] 
  ; [ [6000;6499] ] ; [ [6500;6999] ] ; [ [7000;7499] ] ; [ [7500;7999] ] 
  ; [ [8000;8499] ] ; [ [8500;8999] ] ; [ [9000;9499] ] ; [ [9500;9999] ]
  ] 
  in
  let calc_accu s1 = 
    let imgs1 = Dense.Ndarray.S.get_slice s1 imgs in
    let m = Dense.Ndarray.S.nth_dim imgs1 0 in
    let label1 = Dense.Ndarray.S.get_slice s1 labels in
    let fact1 = mat2num label1 m in
    let pred1 = mat2num (Graph.model model imgs1) m in
    let accu1 = Dense.Matrix.S.(elt_equal pred1 fact1 |> sum') in
    Gc.minor ();
    accu1
  in
  let accu = List.map calc_accu s |> List.fold_left (+.) 0. in
  let m = Dense.Ndarray.S.nth_dim labels 0 in
  let res = (accu /. (float_of_int (m))) in
  write_float_to_file "result.txt" res;
  Owl_log.info "Accuracy on test set: %f" res;;

let train () =
  (* let x, _, y = Dataset.load_mnist_train_data_arr () in *)
  let x, _, y = Dataset.load_cifar_train_data 1 in
  
  let r = Array.init (Owl_dense_ndarray.S.nth_dim x 0) (fun i -> i) in
  let r = Owl_stats.shuffle r in
  
  (* Validation data *)
  let v_rows = Array.sub r 0 2000 in
  let vx = Arr (Owl_dense_ndarray.S.get_fancy [L (Array.to_list v_rows)] x) in
  let vy = Arr (Owl_dense_ndarray.S.rows y v_rows) in

  (* Training data *)
  let t_rows = Array.sub r 2000 8000 in
  let x = Owl_dense_ndarray.S.get_fancy [L (Array.to_list t_rows)] x in
  let y = Owl_dense_ndarray.S.rows y t_rows in

  (* let network = make_network [|28;28;1|] in *)
  let network = make_network [|32;32;3|] in

  (* Hotfix. TODO: Refactor val_loss calculation in owl_optimise_generic *)
  let val_params = Params.config
    ~batch:(Batch.Mini 128) ~learning_rate:(Learning_Rate.Adagrad 0.001) 120.0
  in

  let lowest_val_loss = ref 0. in
  let patience = ref 0 in

  (* define checkpoint function *)
  let chkpt state =
    let open Checkpoint in
    if state.current_batch mod 1 = 0 then (
(*     Owl_log.info "Plotting loss function..";   
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
      let z = Array.map unpack_flt state.loss in
      let c = Array.sub z 0 state.current_batch in 
      write_float_to_file "loss.txt" (c.((Array.length c) - 1));
      write_float_to_file "time.txt" (Unix.gettimeofday () -. state.start_at);

      let _ = match state.current_batch mod state.batches_per_epoch = 0 with
      | false -> ()
      | true  -> let vl = validate_model val_params network (state.current_batch / state.batches_per_epoch - 1) vx vy in
                 write_float_to_file "val_loss.txt" vl;
                 test network;
                 match !lowest_val_loss <> 0. && vl >= !lowest_val_loss with
                      | true  ->  patience := !patience + 1
                      | false ->  lowest_val_loss := vl;
                                  patience := 0
      in
      match !patience >= 50 with 
      | false -> ()
      | true  -> Owl_log.info "Early stopping..";
                 state.stop <- true

    )
  in

  (* plug in chkpt into params *)
  let params = Params.config
    ~batch:(Batch.Mini 128) ~learning_rate:(Learning_Rate.Adagrad 0.001)
    ~checkpoint:(Checkpoint.Custom chkpt) ~stopping:(Stopping.Const 1e-6) 150.0
  in
  Graph.train ~params network x y |> ignore;
  (* (* keep restarting the optimisation until it finishes *)
  let state = Graph.train ~params network x y in
  while Checkpoint.(state.current_batch < state.batches) do
    Checkpoint.(state.stop <- false);
    Graph.train ~state ~params ~init_model:false network x y |> ignore
  done;
 *)
  network

let _ = train () |> test
