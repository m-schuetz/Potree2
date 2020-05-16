
// based on cube example from https://github.com/cx20/webgpu-test (MIT license)

import {LASLoader} from "./LASLoader.js";
import {WebGpuRenderer} from "./src/renderer/WebGpuRenderer.js";
import {PotreeLoader} from "./src/octree/PotreeLoader.js"
import {Camera} from "./src/scene/Camera.js";
import {Quaternion} from "./src/math/Quaternion.js";
import {Matrix4} from "./src/math/Matrix4.js";
import {Vector3} from "./src/math/Vector3.js";
import {OrbitControls} from "./src/navigation/OrbitControls.js";
import {Scene} from "./src/scene/Scene.js";

let canvas = document.getElementById("canvas");
let frameCount = 0;
let lastFpsMeasure = 0;

let renderer = null;
let sceneObject = null;

export let camera = new Camera();
let quaternion = new Quaternion(0, 0, 0, 1);
let controls = new OrbitControls(canvas, camera);

window.quaternion = quaternion;
window.controls = controls;

export let scene = new Scene();

function update(timestamp, delta){
	controls.update(delta);

	scene.update(timestamp, delta);

}

function render(timestamp){

	let {canvas} = renderer;

	renderer.render(scene, camera, sceneObject);


	{// compute FPS
		frameCount++;
		let timeSinceLastFpsMeasure = (performance.now() - lastFpsMeasure) / 1000;
		if(timeSinceLastFpsMeasure > 1){
			let fps = frameCount / timeSinceLastFpsMeasure;
			// console.log(`fps: ${Math.round(fps)}`);
			document.title = `fps: ${Math.round(fps)}`;
			lastFpsMeasure = performance.now();
			frameCount = 0;
		}
	}
}

let previousTimestamp = 0;

function loop(timestamp){

	let delta = timestamp - previousTimestamp;

	let state = {
		timestamp: timestamp,
		delta: delta,
		drawBoundingBox: renderer.drawBoundingBox.bind(renderer),
	};

	update(state);
	render(timestamp, delta);

	requestAnimationFrame(loop);
}

async function run(){

	renderer = await WebGpuRenderer.create(canvas);

	loop();

}

run();