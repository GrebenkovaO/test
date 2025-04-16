import torch
import os
import shutil
import tempfile
import argparse
import gradio as gr
import sys
import io
from PIL import Image
import numpy as np
from source.utils_aux import set_seed
from source.utils_preprocess import read_video_frames, preprocess_frames, select_optimal_frames, save_frames_to_scene_dir, run_colmap_on_scene
from source.trainer import EDGSTrainer
from hydra import initialize, compose
import hydra
from omegaconf import OmegaConf
import time
from source.visualization import generate_circular_camera_path, save_numpy_frames_as_mp4, generate_fully_smooth_cameras_with_tsp, put_text_on_image


# Init RoMA model:
sys.path.append('../submodules/RoMa')
from romatch import roma_outdoor, roma_indoor

roma_model = roma_indoor(device="cuda:0")
roma_model.upsample_preds = False
roma_model.symmetric = False


STATIC_FILE_SERVING_FOLDER = "./served_files"
MODEL_PATH = None
os.makedirs(STATIC_FILE_SERVING_FOLDER, exist_ok=True)

trainer = None

# Capture logs
def capture_logs(func, *args, **kwargs):
    log_capture_string = io.StringIO()
    sys.stdout = log_capture_string
    result = func(*args, **kwargs)
    sys.stdout = sys.__stdout__
    return result, log_capture_string.getvalue()

# Training Pipeline
def run_training_pipeline(scene_dir, 
                          num_ref_views=16, 
                          num_corrs_per_view=20000, 
                          num_steps=1_000,
                          mode_toggle="Ours (EDGS)"):
    with initialize(config_path="./configs", version_base="1.1"):
        cfg = compose(config_name="train")

    scene_name = os.path.basename(scene_dir)
    model_output_dir = f"./outputs/{scene_name}_trained"

    cfg.wandb.mode = "disabled"
    cfg.gs.dataset.model_path = model_output_dir
    cfg.gs.dataset.source_path = scene_dir
    cfg.gs.dataset.images = "images"

    cfg.gs.opt.TEST_CAM_IDX_TO_LOG = 12
    cfg.train.gs_epochs = 30000
    
    if mode_toggle=="Ours (EDGS)":
        cfg.gs.opt.opacity_reset_interval = 1_000_000
        cfg.train.reduce_opacity = True
        cfg.train.no_densify = True
        cfg.train.max_lr = True

        cfg.init_wC.use = True
        cfg.init_wC.matches_per_ref = num_corrs_per_view
        cfg.init_wC.nns_per_ref = 2
        cfg.init_wC.num_refs = num_ref_views
        cfg.init_wC.add_SfM_init = False
        cfg.init_wC.scaling_factor = 0.00077 * 2.
    else:
        cfg.gs.opt.opacity_reset_interval = 3_000
        cfg.train.reduce_opacity = False
        cfg.train.no_densify = False
        cfg.train.max_lr = False

        cfg.init_wC.use = False
        
    set_seed(cfg.seed)
    os.makedirs(cfg.gs.dataset.model_path, exist_ok=True)

    global trainer
    global MODEL_PATH
    generator3dgs = hydra.utils.instantiate(cfg.gs, do_train_test_split=False)
    trainer = EDGSTrainer(GS=generator3dgs, training_config=cfg.gs.opt, device=cfg.device, log_wandb=cfg.wandb.mode != 'disabled')

    # Disable evaluation and saving
    trainer.saving_iterations = []
    trainer.evaluate_iterations = []

    # Initialize
    trainer.timer.start()
    start_time = time.time()
    trainer.init_with_corr(cfg.init_wC, roma_model=roma_model)
    time_for_init = time.time()-start_time

    viewpoint_cams = trainer.GS.scene.getTrainCameras()
    path_cameras = generate_fully_smooth_cameras_with_tsp(existing_cameras=viewpoint_cams, 
                                                          n_selected=8, 
                                                          n_points_per_segment=30, 
                                                          closed=True)

    path_renderings = []
    idx = 0
    # Visualize after init
    for _ in range(120):
        with torch.no_grad():
            viewpoint_cam = path_cameras[idx]
            idx = (idx + 1) % len(path_cameras)
            render_pkg = trainer.GS(viewpoint_cam)
            image = render_pkg["render"]
            image_np = np.clip(image.detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            image_np = (image_np * 255).astype(np.uint8)
            path_renderings.append(put_text_on_image(img=image_np, 
                                                     text=f"Init stage.\nTime:{time_for_init:.3f}s.   "))
    path_renderings = path_renderings + [put_text_on_image(img=image_np, text=f"Start fitting.\nTime:{time_for_init:.3f}s.   ")]*30
    
    # Train and save visualizations during training.
    start_time = time.time()
    for _ in range(int(num_steps//10)):
        with torch.no_grad():
            viewpoint_cam = path_cameras[idx]
            idx = (idx + 1) % len(path_cameras)
            render_pkg = trainer.GS(viewpoint_cam)
            image = render_pkg["render"]
            image_np = np.clip(image.detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            image_np = (image_np * 255).astype(np.uint8)
            path_renderings.append(put_text_on_image(
                img=image_np, 
                text=f"Fitting stage.\nTime:{time_for_init + time.time()-start_time:.3f}s.   "))
    
        cfg.train.gs_epochs = 10
        trainer.train(cfg.train)
        print(f"Time elapsed: {(time_for_init + time.time()-start_time):.2f}s.")
        # if (cfg.init_wC.use == False) and (time_for_init + time.time()-start_time) > 60:
        #     break
    final_time = time.time()
    
    # Add static frame. To highlight we're done
    path_renderings += [put_text_on_image(
        img=image_np, text=f"Done.\nTime:{time_for_init + final_time -start_time:.3f}s.   ")]*30
    # Final rendering at the end.
    for _ in range(len(path_cameras)):
        with torch.no_grad():
            viewpoint_cam = path_cameras[idx]
            idx = (idx + 1) % len(path_cameras)
            render_pkg = trainer.GS(viewpoint_cam)
            image = render_pkg["render"]
            image_np = np.clip(image.detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            image_np = (image_np * 255).astype(np.uint8)
            path_renderings.append(put_text_on_image(img=image_np, 
                                                 text=f"Final result.\nTime:{time_for_init + final_time -start_time:.3f}s.   "))

    trainer.save_model()
    final_video_path = os.path.join(STATIC_FILE_SERVING_FOLDER, f"{scene_name}_final.mp4")
    save_numpy_frames_as_mp4(frames=path_renderings, output_path=final_video_path, fps=30, center_crop=0.85)
    MODEL_PATH = cfg.gs.dataset.model_path
    ply_path = os.path.join(cfg.gs.dataset.model_path, f"point_cloud/iteration_{trainer.gs_step}/point_cloud.ply")
    shutil.copy(ply_path, os.path.join(STATIC_FILE_SERVING_FOLDER, "point_cloud_final.ply"))

    return final_video_path, ply_path

# Gradio Interface
def gradio_interface(input_path, num_ref_views, num_corrs, num_steps, mode_toggle):
    images, scene_dir = run_full_pipeline(input_path, num_ref_views, num_corrs)
    (final_video_path, ply_path), log_output = capture_logs(run_training_pipeline,
                                                            scene_dir,
                                                            num_ref_views,
                                                            num_corrs,
                                                            num_steps,
                                                            mode_toggle)
    images_rgb = [img[:, :, ::-1] for img in images]
    return images_rgb, final_video_path, scene_dir, ply_path, log_output

# Dummy Render Functions
def render_all_views(scene_dir):
    viewpoint_cams = trainer.GS.scene.getTrainCameras()
    path_cameras = generate_fully_smooth_cameras_with_tsp(existing_cameras=viewpoint_cams, 
                                                          n_selected=8, 
                                                          n_points_per_segment=60, 
                                                          closed=False)
    path_cameras = path_cameras + path_cameras[::-1]

    path_renderings = []
    with torch.no_grad():
        for viewpoint_cam in path_cameras:
            render_pkg = trainer.GS(viewpoint_cam)
            image = render_pkg["render"]
            image_np = np.clip(image.detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            image_np = (image_np * 255).astype(np.uint8)
            path_renderings.append(image_np)
    save_numpy_frames_as_mp4(frames=path_renderings, 
                             output_path=os.path.join(STATIC_FILE_SERVING_FOLDER, "render_all_views.mp4"), 
                             fps=30, 
                             center_crop=0.85)
    
    return os.path.join(STATIC_FILE_SERVING_FOLDER, "render_all_views.mp4")

def render_circular_path(scene_dir):
    viewpoint_cams = trainer.GS.scene.getTrainCameras()
    path_cameras = generate_circular_camera_path(existing_cameras=viewpoint_cams, 
                                                 N=240, 
                                                 radius_scale=0.65,
                                                 d=0)

    path_renderings = []
    with torch.no_grad():
        for viewpoint_cam in path_cameras:
            render_pkg = trainer.GS(viewpoint_cam)
            image = render_pkg["render"]
            image_np = np.clip(image.detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            image_np = (image_np * 255).astype(np.uint8)
            path_renderings.append(image_np)
    save_numpy_frames_as_mp4(frames=path_renderings, 
                             output_path=os.path.join(STATIC_FILE_SERVING_FOLDER, "render_circular_path.mp4"), 
                             fps=30, 
                             center_crop=0.85)
    
    return os.path.join(STATIC_FILE_SERVING_FOLDER, "render_circular_path.mp4")

# Download Functions
def download_cameras():
    path = os.path.join(MODEL_PATH, "cameras.json")
    return f"[📥 Download Cameras.json](file={path})"

def download_model():
    path = os.path.join(STATIC_FILE_SERVING_FOLDER, "point_cloud_final.ply")
    return f"[📥 Download Pretrained Model (.ply)](file={path})"

# Full pipeline helpers
def run_full_pipeline(input_path, num_ref_views, num_corrs):
    tmpdirname = tempfile.mkdtemp()
    scene_dir = os.path.join(tmpdirname, "scene")
    os.makedirs(scene_dir, exist_ok=True)

    selected_frames = process_input(input_path, num_ref_views, scene_dir)
    run_colmap_on_scene(scene_dir)

    return selected_frames, scene_dir

# Preprocess Input
def process_input(input_path, num_ref_views, output_dir):
    if isinstance(input_path, (str, os.PathLike)):
        if os.path.isdir(input_path):
            frames = []
            for img_file in sorted(os.listdir(input_path)):
                if img_file.lower().endswith(('jpg', 'jpeg', 'png')):
                    img = Image.open(os.path.join(output_dir, img_file)).convert('RGB')
                    img.thumbnail((1024, 1024))
                    frames.append(np.array(img))
        else:
            frames = read_video_frames(video_input=input_path)
    else:
        frames = read_video_frames(video_input=input_path)

    frames_scores = preprocess_frames(frames)
    selected_frames_indices = select_optimal_frames(scores=frames_scores, k=num_ref_views)
    selected_frames = [frames[frame_idx] for frame_idx in selected_frames_indices]

    save_frames_to_scene_dir(frames=selected_frames, scene_dir=output_dir)
    return selected_frames

# Gradio App
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=6):
            gr.Markdown("""
            ## <span style='font-size: 20px;'>📄 EDGS: Eliminating Densification for Efficient Convergence of 3DGS</span>
            🔗 <a href='https://compvis.github.io/EDGS' target='_blank'>Project Page</a>
            """, elem_id="header")

    gr.Markdown("""
    ### <span style='font-size: 22px;'>🛠️ How to Use This Demo</span>

    1. Upload a **front-facing video** or **a folder of images** of a **static** scene.
    2. Use the sliders to configure the number of reference views, correspondences, and optimization steps.
    3. Click **🚀 Start Reconstruction** to launch the pipeline.
    4. Watch the training visualization and explore the 3D model.
    ‼️ **If you see nothing in the 3D model viewer**, try rotating or zooming — sometimes the initial camera orientation is off.


    ✅ Best for scenes with small camera motion.
    ❗ For full 360° or large-scale scenes, we recommend the Colab version (see project page).
    """, elem_id="quickstart")


    with gr.Tabs():
        with gr.TabItem("Training Pipeline"):
            scene_dir_state = gr.State()
            ply_model_state = gr.State()
            with gr.Row(visible=False) as spinner:
                gr.Markdown("⏳ Processing, please wait...")
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📥 Upload Input")
                    input_file = gr.File(label="Upload Video or Images", 
                        file_types=[".mp4", ".avi", ".mov", ".png", ".jpg", ".jpeg"], 
                        file_count="multiple")
                    ref_slider = gr.Slider(4, 32, value=16, step=1, label="Number of Reference Views")
                    corr_slider = gr.Slider(5000, 20000, value=30000, step=1000, label="Correspondences per Reference View")
                    fit_steps_slider = gr.Slider(200, 5000, value=400, step=200, label="Number of optimization steps")
                    mode_toggle = gr.Radio(["Ours (EDGS)", "Vanilla 3DGS"], label="Select Method", value="Ours (EDGS)")
                    start_button = gr.Button("🚀 Start Reconstruction")
                    gallery = gr.Gallery(label="Selected Reference Views", columns=4, height=300)
                with gr.Column(scale=3):
                    gr.Markdown("### 🏋️ Training Visualization")
                    video_output = gr.Video(label="Training Video", autoplay=True)
                    render_all_views_button = gr.Button("🎥 Render All-Views Path")
                    render_circular_path_button = gr.Button("🎥 Render Circular Path")
                    rendered_video_output = gr.Video(label="Rendered Video", autoplay=True)
                with gr.Column(scale=5):
                    gr.Markdown("### 🌐 Final 3D Model")
                    model3d_viewer = gr.Model3D(label="3D Model Viewer")

                    gr.Markdown("### 📦 Output Files")
                    with gr.Row(height=50):
                        with gr.Column():
                            #gr.Markdown(value=f"[📥 Download .ply](file/point_cloud_final.ply)")
                            download_cameras_button = gr.Button("📥 Download Cameras.json")
                            download_cameras_file = gr.File(label="📄 Cameras.json")
                        with gr.Column():
                            download_model_button = gr.Button("📥 Download Pretrained Model (.ply)")
                            download_model_file = gr.File(label="📄 Pretrained Model (.ply)")
            log_output_box = gr.Textbox(label="🖥️ Log", lines=10, interactive=False)

    gr.Markdown("""
    ---
    ### <span style='font-size: 20px;'>📖 Detailed Overview</span>

    If you uploaded a video, it will be automatically cut into a smaller number of frames (default: 16).

    The model pipeline:
    1. 🧠 Runs PyCOLMAP to estimate camera intrinsics & poses (~3–7 seconds for <16 images).
    2. 🔁 Computes 2D-2D correspondences between views. More correspondences generally improve quality.
    3. 🔧 Optimizes a 3D Gaussian Splatting model for several steps.

    ### 🎥 Training Visualization
    You will see a visualization of the entire training process in the "Training Video" pane.

    ### 🌀 Rendering & 3D Model
    - Render the scene from a circular path of novel views.
    - Or from camera views close to the original input.

    The 3D model is shown in the right viewer. You can explore it interactively:
    - On PC: WASD keys, arrow keys, and mouse clicks
    - On mobile: pan and pinch to zoom

    🕒 Note: the 3D viewer takes a few extra seconds (~5s) to display after training ends.

    ---
    Preloaded models coming soon. (TODO)
    """, elem_id="details")

    start_button.click(
        fn=gradio_interface,
        preprocess=lambda: spinner.update(visible=True),
        inputs=[input_file, ref_slider, corr_slider, fit_steps_slider, mode_toggle],
        outputs=[gallery, video_output, scene_dir_state, model3d_viewer, log_output_box],
        postprocess=lambda: spinner.update(visible=False)
    )

    render_all_views_button.click(fn=render_all_views, inputs=[scene_dir_state], outputs=[rendered_video_output])
    render_circular_path_button.click(fn=render_circular_path, inputs=[scene_dir_state], outputs=[rendered_video_output])

    download_cameras_button.click(fn=lambda: os.path.join(MODEL_PATH, "cameras.json"), inputs=[], outputs=[download_cameras_file])
    download_model_button.click(fn=lambda: os.path.join(STATIC_FILE_SERVING_FOLDER, "point_cloud_final.ply"), inputs=[], outputs=[download_model_file])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Gradio demo for EDGS preprocessing and 3D viewing.")
    parser.add_argument("--port", type=int, default=7860, help="Port to launch the Gradio app on.")
    parser.add_argument("--no_share", action='store_true', help="Disable Gradio sharing and assume local access (default: share=True)")
    args = parser.parse_args()

    demo.launch(server_name="0.0.0.0", server_port=args.port, share=not args.no_share)
