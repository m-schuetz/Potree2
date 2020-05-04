
// based on cube example from https://github.com/cx20/webgpu-test (MIT license)

import {LASLoader} from "./LASLoader.js";
import {WebGpuRenderer} from "./WebGpuRenderer.js";
import {PotreeLoader} from "./src/octree/PotreeLoader.js"
import {Camera} from "./src/scene/Camera.js";
import {Quaternion} from "./src/math/Quaternion.js";
import {Matrix4} from "./src/math/Matrix4.js";
import {Vector3} from "./src/math/Vector3.js";
import {OrbitControls} from "./src/navigation/OrbitControls.js";


let urlPotree = "http://localhost:8080/nocommit/lion/metadata.json";

let canvas = document.getElementById("canvas");
let frameCount = 0;
let lastFpsMeasure = 0;

let renderer = null;
let sceneObject = null;
let worldViewProj = mat4.create();

export let camera = new Camera();
let quaternion = new Quaternion(0, 0, 0, 1);
let controls = new OrbitControls(canvas, camera);

window.quaternion = quaternion;
window.controls = controls;

export let scene = {
	meshes: [],
	nodes: [],
};

async function initScene(){

	// let pointcloud = await PotreeLoader.load(urlPotree);
	// await pointcloud.loader.loadHierarchy(pointcloud.root);
	// await pointcloud.loader.loadNode(pointcloud.root);

	// let node = pointcloud.root;

	// let position = pointcloud.root.buffers.position.buffer;
	// let rgb = pointcloud.root.buffers.rgb.buffer;
	// let numPoints = node.numPoints;

	// let {device} = renderer;

	// let [bufPositions, posMapping] = device.createBufferMapped({
	// 	size: 12 * numPoints,
	// 	usage: GPUBufferUsage.VERTEX,
	// });
	// new Int32Array(posMapping).set(new Int32Array(position));
	// bufPositions.unmap();

	// let [bufRGB, mappingRGB] = device.createBufferMapped({
	// 	size: 4 * numPoints,
	// 	usage: GPUBufferUsage.VERTEX,
	// });
	// new Uint8Array(mappingRGB).set(new Uint8Array(rgb));
	// bufRGB.unmap();

	// sceneObject = {
	// 	n: numPoints,
	// 	bufPositions: bufPositions,
	// 	bufColors: bufRGB,
	// };

}


function update(timestamp, delta){

	let {canvas} = renderer;

	controls.update(delta);

	{ // update worldViewProj
		let proj = mat4.create();
		let view = mat4.create();

		{ // proj
			const aspect = Math.abs(canvas.width / canvas.height);
			mat4.perspective(proj, 45, aspect, 0.1, 100.0);
		}

		{ // view
			let position = controls.getPosition();
			let target = controls.target;
			let up = [0, 0, 1];
			mat4.lookAt(view, position.toArray(), target.toArray(), up);

		}

		mat4.multiply(worldViewProj, proj, view);
	}

}

function render(timestamp){

	let {canvas} = renderer;

	renderer.render(scene.nodes, camera, sceneObject);


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

	update(timestamp, delta);
	render(timestamp, delta);

	requestAnimationFrame(loop);
}

async function run(){

	renderer = await WebGpuRenderer.create(canvas);

	await initScene();

	loop();

}

run();