import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gradio as gr
import numpy as np
import random
import torch
import gc
import json
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline

gc.collect()
torch.cuda.empty_cache()

MAX_SEED = np.iinfo(np.int32).max
dtype = torch.bfloat16

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=dtype,
    low_cpu_mem_usage=False
)

pipe.load_lora_weights("lightx2v/Qwen-Image-Edit-2511-Lightning", weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", adapter_name="lightning")
pipe.load_lora_weights("fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA", weight_name="qwen-image-edit-2511-multiple-angles-lora.safetensors", adapter_name="angles")
pipe.set_adapters(["lightning", "angles"], adapter_weights=[1.0, 1.0])
pipe.enable_model_cpu_offload()

AZIMUTH_MAP = {0: "front view", 45: "front-right quarter view", 90: "right side view", 135: "back-right quarter view", 180: "back view", 225: "back-left quarter view", 270: "left side view", 315: "front-left quarter view"}
ELEVATION_MAP = {-30: "low-angle shot", 0: "eye-level shot", 30: "elevated shot", 60: "high-angle shot"}
DISTANCE_MAP = {0.6: "close-up", 1.0: "medium shot", 1.4: "medium shot", 1.8: "wide shot"}

def snap_to_nearest(value, options):
    return min(options, key=lambda x: abs(x - value))

def build_camera_prompt(azimuth, elevation, distance):
    az_s = snap_to_nearest(azimuth, list(AZIMUTH_MAP.keys()))
    el_s = snap_to_nearest(elevation, list(ELEVATION_MAP.keys()))
    di_s = snap_to_nearest(distance, list(DISTANCE_MAP.keys()))
    return f"<sks> {AZIMUTH_MAP[az_s]} {ELEVATION_MAP[el_s]} {DISTANCE_MAP[di_s]}"

def infer_camera_edit(image, azimuth, elevation, distance, seed, randomize_seed, guidance_scale, num_inference_steps, height, width, progress=gr.Progress()):
    if image is None: raise gr.Error("Upload image.")
    prompt = build_camera_prompt(azimuth, elevation, distance)
    if randomize_seed: seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    result = pipe(
        image=[image.convert("RGB")],
        prompt=prompt,
        height=int(height),
        width=int(width),
        num_inference_steps=int(num_inference_steps),
        generator=generator,
        guidance_scale=float(guidance_scale),
        num_images_per_prompt=1,
    ).images[0]
    return result, seed, prompt

def update_dimensions_on_upload(image):
    if image is None: return 1024, 1024
    w, h = image.size
    nw, nh = (1024, int(1024*(h/w))) if w > h else (int(1024*(w/h)), 1024)
    return max(256, (nw // 8) * 8), max(256, (nh // 8) * 8)

def update_3d_img(img):
    if img is None: return ""
    import base64; from io import BytesIO
    buf = BytesIO(); img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

def parse_3d_bridge(json_str):
    try:
        data = json.loads(json_str)
        az = snap_to_nearest(data.get('azimuth', 0), list(AZIMUTH_MAP.keys()))
        el = snap_to_nearest(data.get('elevation', 0), list(ELEVATION_MAP.keys()))
        di = snap_to_nearest(data.get('distance', 1.0), list(DISTANCE_MAP.keys()))
        return az, el, di
    except:
        return 0, 0, 1.0

def update_js_bridge(az, el, dist):
    return json.dumps({"azimuth": az, "elevation": el, "distance": dist})

css = """
#camera-control-wrapper { width: 100%; height: 450px; position: relative; background: #111; border-radius: 8px; border: 1px solid #333; overflow: hidden; }
#prompt-overlay { position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.8); padding: 6px 12px; border-radius: 6px; font-family: monospace; font-size: 12px; color: #00ff88; white-space: nowrap; z-index: 10; pointer-events: none; border: 1px solid #444; }
canvas { display: block; width: 100% !important; height: 100% !important; }
#js_bridge, #img_bridge { display: none !important; } 
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 Qwen Camera Edit 2511")
    
    js_bridge = gr.Textbox(value='{"azimuth":0,"elevation":0,"distance":1.0}', elem_id="js_bridge")
    img_bridge = gr.Textbox(elem_id="img_bridge")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input", type="pil")
            gr.HTML(value='<div id="camera-control-wrapper"><div id="prompt-overlay">Initializing Scene...</div></div>')
            run_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
            
            az_s = gr.Slider(0, 315, 45, label="Azimuth", value=0)
            el_s = gr.Slider(-30, 60, 30, label="Elevation", value=0)
            
            di_s = gr.Slider(0.2, 1.8, 0.4, label="Distance", value=1.0) 
            
            prompt_p = gr.Textbox(label="Prompt Preview", interactive=False)
            
        with gr.Column():
            output_img = gr.Image(label="Output")
            with gr.Accordion("Settings", open=False):
                seed = gr.Slider(0, MAX_SEED, value=0, label="Seed")
                rand = gr.Checkbox(label="Random", value=True)
                gs = gr.Slider(1.0, 10.0, value=1.5, label="Guidance")
                steps = gr.Slider(1, 20, value=4, label="Steps")
                h_s = gr.Slider(256, 2048, 8, value=1024, label="Height")
                w_s = gr.Slider(256, 2048, 8, value=1024, label="Width")

    demo.load(None, None, None, js="""
    () => {
        const init = () => {
            const wrapper = document.querySelector('#camera-control-wrapper');
            const overlay = document.querySelector('#prompt-overlay');
            if (!wrapper || typeof THREE === 'undefined') { setTimeout(init, 100); return; }
            
            const scene = new THREE.Scene(); scene.background = new THREE.Color(0x111111);
            const camera = new THREE.PerspectiveCamera(50, wrapper.clientWidth/wrapper.clientHeight, 0.1, 1000);
            camera.position.set(5, 3.5, 5); camera.lookAt(0, 0.75, 0);
            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            wrapper.appendChild(renderer.domElement);
            
            scene.add(new THREE.AmbientLight(0xffffff, 0.7));
            const dl = new THREE.DirectionalLight(0xffffff, 0.8); dl.position.set(5, 10, 5); scene.add(dl);
            scene.add(new THREE.GridHelper(10, 20, 0x333333, 0x222222));
            const CENTER = new THREE.Vector3(0, 0.75, 0);
            
            let az = 0, el = 0, df = 1.0;
            const azS = [0, 45, 90, 135, 180, 225, 270, 315], elS = [-30, 0, 30, 60], diS = [0.6, 1.0, 1.4, 1.8];
            const planeMat = new THREE.MeshBasicMaterial({ color: 0x444444, side: THREE.DoubleSide });
            let plane = new THREE.Mesh(new THREE.PlaneGeometry(1.5, 1.5), planeMat); plane.position.copy(CENTER); scene.add(plane);
            
            const cam = new THREE.Group();
            const body = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.25, 0.45), new THREE.MeshStandardMaterial({color: 0x4477aa}));
            const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.12, 0.25), new THREE.MeshStandardMaterial({color: 0x222}));
            lens.rotation.x = Math.PI/2; lens.position.z = 0.3; cam.add(body, lens); scene.add(cam);
            
            const azRing = new THREE.Mesh(new THREE.TorusGeometry(2.5, 0.03, 16, 100), new THREE.MeshBasicMaterial({color: 0x00ff88, transparent: true, opacity: 0.3}));
            azRing.rotation.x = Math.PI/2; scene.add(azRing);
            const arcPts = []; for(let i=0; i<=32; i++){ let a = THREE.MathUtils.degToRad(-30 + 90*i/32); arcPts.push(new THREE.Vector3(-1.2, 2.0*Math.sin(a)+0.75, 2.0*Math.cos(a))); }
            const elArc = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(arcPts), 32, 0.03, 8, false), new THREE.MeshBasicMaterial({color: 0xff69b4, transparent: true, opacity: 0.3}));
            scene.add(elArc);
            
            const hAz = new THREE.Mesh(new THREE.SphereGeometry(0.2), new THREE.MeshStandardMaterial({color: 0x00ff88, emissive: 0x00ff88}));
            const hEl = new THREE.Mesh(new THREE.SphereGeometry(0.2), new THREE.MeshStandardMaterial({color: 0xff69b4, emissive: 0xff69b4}));
            const hDi = new THREE.Mesh(new THREE.SphereGeometry(0.2), new THREE.MeshStandardMaterial({color: 0xffa500, emissive: 0xffa500}));
            scene.add(hAz, hEl, hDi);
            const line = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color: 0xffa500, transparent: true, opacity: 0.5})); scene.add(line);

            const snap = (v, arr) => arr.reduce((p, c) => Math.abs(c-v) < Math.abs(p-v) ? c : p);
            const update = () => {
                const d = 1.6 * df, aR = THREE.MathUtils.degToRad(az), eR = THREE.MathUtils.degToRad(el);
                cam.position.set(d*Math.sin(aR)*Math.cos(eR), d*Math.sin(eR)+0.75, d*Math.cos(aR)*Math.cos(eR));
                cam.lookAt(CENTER);
                hAz.position.set(2.5*Math.sin(aR), 0, 2.5*Math.cos(aR));
                hEl.position.set(-1.2, 2.0*Math.sin(eR)+0.75, 2.0*Math.cos(eR));
                hDi.position.set((d-0.6)*Math.sin(aR)*Math.cos(eR), (d-0.6)*Math.sin(eR)+0.75, (d-0.6)*Math.cos(aR)*Math.cos(eR));
                line.geometry.setFromPoints([cam.position, CENTER]);
                overlay.innerText = `AZ: ${snap(az, azS)}° | EL: ${snap(el, elS)}° | ZOOM: ${snap(df, diS).toFixed(1)}`;
            };

            const ray = new THREE.Raycaster(); const mouse = new THREE.Vector2(); let active = null; let lastY = 0;
            renderer.domElement.addEventListener('mousedown', (e) => {
                const r = renderer.domElement.getBoundingClientRect();
                mouse.x = ((e.clientX - r.left)/r.width)*2 - 1; mouse.y = -((e.clientY - r.top)/r.height)*2 + 1;
                ray.setFromCamera(mouse, camera); const hits = ray.intersectObjects([hAz, hEl, hDi]);
                if(hits.length) { active = hits[0].object; active.scale.setScalar(1.3); lastY = mouse.y; }
            });
            window.addEventListener('mousemove', (e) => {
                if(!active) return;
                const r = renderer.domElement.getBoundingClientRect();
                mouse.x = ((e.clientX - r.left)/r.width)*2 - 1; mouse.y = -((e.clientY - r.top)/r.height)*2 + 1;
                ray.setFromCamera(mouse, camera);
                if(active === hAz) {
                    const p = new THREE.Plane(new THREE.Vector3(0,1,0), 0); const i = new THREE.Vector3();
                    if(ray.ray.intersectPlane(p, i)) az = (THREE.MathUtils.radToDeg(Math.atan2(i.x, i.z)) + 360) % 360;
                } else if(active === hEl) {
                    const p = new THREE.Plane(new THREE.Vector3(1,0,0), 1.2); const i = new THREE.Vector3();
                    if(ray.ray.intersectPlane(p, i)) el = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(Math.atan2(i.y-0.75, i.z)), -30, 60);
                } else if(active === hDi) { df = THREE.MathUtils.clamp(df - (mouse.y - lastY)*2, 0.6, 1.8); }
                lastY = mouse.y; update();
            });
            window.addEventListener('mouseup', () => {
                if(active) {
                    active.scale.setScalar(1);
                    az = snap(az, azS); el = snap(el, elS); df = snap(df, diS);
                    const bridge = document.querySelector('#js_bridge textarea');
                    if(bridge) { bridge.value = JSON.stringify({azimuth: az, elevation: el, distance: df}); bridge.dispatchEvent(new Event('input')); }
                }
                active = null; update();
            });
            setInterval(() => {
                const b = document.querySelector('#js_bridge textarea');
                if(b && b.value !== lastV) {
                    lastV = b.value;
                    try { const v = JSON.parse(b.value); az = v.azimuth; el = v.elevation; df = v.distance; update(); } catch(e){}
                }
                const i = document.querySelector('#img_bridge textarea');
                if(i && i.value && i.value !== lastI) {
                    lastI = i.value;
                    new THREE.TextureLoader().load(lastI, t => {
                        t.colorSpace = THREE.SRGBColorSpace;
                        planeMat.map = t; planeMat.needsUpdate = true;
                        const ar = t.image.width / t.image.height;
                        plane.geometry.dispose();
                        plane.geometry = new THREE.PlaneGeometry(ar > 1 ? 1.5 : 1.5*ar, ar > 1 ? 1.5/ar : 1.5);
                    });
                }
            }, 200);
            let lastV = "", lastI = "";
            new ResizeObserver(() => { renderer.setSize(wrapper.clientWidth, wrapper.clientHeight); camera.aspect = wrapper.clientWidth/wrapper.clientHeight; camera.updateProjectionMatrix(); }).observe(wrapper);
            update(); (function run(){ requestAnimationFrame(run); renderer.render(scene, camera); })();
        };
        init();
    }
    """)

    js_bridge.change(parse_3d_bridge, js_bridge, [az_s, el_s, di_s]).then(build_camera_prompt, [az_s, el_s, di_s], prompt_p)
    
    for s in [az_s, el_s, di_s]:
        s.release(update_js_bridge, [az_s, el_s, di_s], js_bridge)
        s.change(build_camera_prompt, [az_s, el_s, di_s], prompt_p)

    run_btn.click(infer_camera_edit, [input_img, az_s, el_s, di_s, seed, rand, gs, steps, h_s, w_s], [output_img, seed, prompt_p])
    input_img.upload(update_dimensions_on_upload, input_img, [w_s, h_s]).then(update_3d_img, input_img, img_bridge)

if __name__ == "__main__":
    demo.queue().launch(share=False, head='<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>')