
import {EventDispatcher} from "./EventDispatcher.js";
import {WebGpuRenderer} from "./renderer/WebGpuRenderer.js";
import {Camera} from "./scene/Camera.js";
import {OrbitControls} from "./navigation/OrbitControls.js";
import {Scene} from "./scene/Scene.js";

export class Potree extends EventDispatcher{

	constructor(canvas){
		super();
	}

	async init(canvas){
		this.canvas = canvas;
		this.frameCount = 0;
		this.lastFpsMeasure = 0;
		this.previousTimestamp = 0;
		this.camera = new Camera();
		this.scene = new Scene();
		this.controls = new OrbitControls(canvas, this.camera);

		this.renderer = await WebGpuRenderer.create(canvas);

		this.loop();

		return this;
	}

	loop(timestamp){
		let delta = timestamp - this.previousTimestamp;
		let {renderer} = this;

		let state = {
			timestamp: timestamp,
			delta: delta,
			drawBoundingBox: renderer.drawBoundingBox.bind(renderer),
			scene: this.scene,
			camera: this.camera,
		};

		this.update(state);
		this.render(timestamp, delta);

		this.previousTimestamp = timestamp;

		requestAnimationFrame(this.loop.bind(this));
	}

	update(timestamp, delta){
		let {scene, controls} = this;

		this.dispatch("update", {
			timestamp: timestamp, 
			delta: delta,
		});

		controls.update(delta);

		scene.update(timestamp, delta);
	}

	render(){
		let {renderer, scene, camera} = this;

		renderer.render(scene, camera);

		{// compute FPS
			this.frameCount++;
			let timeSinceLastFpsMeasure = (performance.now() - this.lastFpsMeasure) / 1000;
			if(timeSinceLastFpsMeasure > 1){
				let fps = this.frameCount / timeSinceLastFpsMeasure;
				// console.log(`fps: ${Math.round(fps)}`);
				document.title = `fps: ${Math.round(fps)}`;
				this.lastFpsMeasure = performance.now();
				this.frameCount = 0;
				this.fps = fps;
			}
		}
	}

}
