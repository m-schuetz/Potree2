
// based on cube example from https://github.com/cx20/webgpu-test (MIT license)

import {LASLoader} from "./LASLoader.js";
import {WebGpuRenderer} from "./WebGpuRenderer.js";
import {PotreeLoader} from "./src/octree/PotreeLoader.js"
import {Camera} from "./src/scene/Camera.js";
import {Quaternion} from "./src/math/Quaternion.js";
import {Matrix4} from "./src/math/Matrix4.js";
import {Vector3} from "./src/math/Vector3.js";
import {OrbitControls} from "./src/navigation/OrbitControls.js";
import {createTestMesh} from "./src/Mesh.js";


let urlPotree = "http://localhost:8080/nocommit/lion/metadata.json";

let canvas = document.getElementById("canvas");
let frameCount = 0;
let lastFpsMeasure = 0;

let renderer = null;
let sceneObject = null;
let testMesh = null;
let worldViewProj = mat4.create();

let camera = new Camera();
let quaternion = new Quaternion(0, 0, 0, 1);
let controls = new OrbitControls(canvas, camera);

window.quaternion = quaternion;
window.controls = controls;

async function initScene(){

	let pointcloud = await PotreeLoader.load(urlPotree);
	await pointcloud.loader.loadHierarchy(pointcloud.root);
	await pointcloud.loader.loadNode(pointcloud.root);

	let node = pointcloud.root;

	let position = pointcloud.root.buffers.position.buffer;
	let rgb = pointcloud.root.buffers.rgb.buffer;
	let numPoints = node.numPoints;

	let {device} = renderer;

	let [bufPositions, posMapping] = device.createBufferMapped({
		size: 12 * numPoints,
		usage: GPUBufferUsage.VERTEX,
	});
	new Int32Array(posMapping).set(new Int32Array(position));
	bufPositions.unmap();

	let [bufRGB, mappingRGB] = device.createBufferMapped({
		size: 4 * numPoints,
		usage: GPUBufferUsage.VERTEX,
	});
	new Uint8Array(mappingRGB).set(new Uint8Array(rgb));
	bufRGB.unmap();

	sceneObject = {
		n: numPoints,
		bufPositions: bufPositions,
		bufColors: bufRGB,
	};

	

	testMesh = createTestMesh(renderer);
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
			//let target = vec3.fromValues(2, 5, 0);
			// let target = vec3.fromValues(0, 0, 0);
			// let r = 10;
			// let x = r * Math.sin(timestamp / 1000) + target[0];
			// let y = r * Math.cos(timestamp / 1000) + target[1];
			// let z = 2;

			// let position = vec3.fromValues(x, y, z);
			// let up = vec3.fromValues(0, 0, 1);
			// mat4.lookAt(view, position, target, up);

			//quaternion.x = controls.yaw;

			// let qYaw = new Quaternion().setFromEuler(0, 0, controls.yaw);
			// let qPitch = new Quaternion().setFromEuler(0, 0, 0);
			// let qOrientation = new Quaternion().multiplyQuaternions(qPitch, qYaw);

			let qOrientation = new Quaternion().setFromEuler(0, 0, controls.yaw);

			//quaternion.setFromEuler(0, 0, controls.yaw);
			// rotate.setFromQuaternion(qOrientation);
			let rotate = new Matrix4().makeRotationZ(controls.yaw);

			let translate = mat4.create();
			
			// mat4.translate(translate, translate, [0, -controls.radius, 0]);
			let campos = controls.getPosition();
			mat4.translate(translate, translate, [campos.x, campos.y, campos.z]);
			
			let tmp = mat4.create();

			mat4.multiply(tmp, translate, rotate.elements);
			//mat4.multiply(tmp, rotate.elements, translate);
			// mat4.invert(tmp, tmp);
			// mat4.multiply(tmp, translate, rotate.elements);#

			let position = controls.getPosition();
			let target = controls.target;
			let up = [0, 0, 1];
			mat4.lookAt(view, position.toArray(), target.toArray(), up);

			// let flip = [
			// 	1, 0, 0, 0,
			// 	0, 0, 1, 0,
			// 	0, 1, 0, 0,
			// 	0, 0, 0, 1,
			// ];
			// mat4.multiply(view, flip, tmp);



			//mat4.multiply(view, translate, rotate);

			// camera.position.set(x, y, z);
			// camera.lookAt(target);
		}

		mat4.multiply(worldViewProj, proj, view);
	}

}

function render(timestamp){

	let {canvas} = renderer;

	let needsResize = canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight;
	if(needsResize){
		canvas.width = canvas.clientWidth;
		canvas.height = canvas.clientHeight;

		renderer.depthTexture = renderer.device.createTexture({
			size: {
				width: canvas.width,
				height: canvas.height,
				depth: 1
			},
			format: "depth24plus-stencil8",
			usage: GPUTextureUsage.OUTPUT_ATTACHMENT
		});
	}

	renderer.uniformBuffer.setSubData(0, worldViewProj);

	const commandEncoder = renderer.device.createCommandEncoder();
	const textureView = renderer.swapChain.getCurrentTexture().createView();
	const renderPassDescriptor = {
		colorAttachments: [{
			attachment: textureView,
			loadValue: { r: 0, g: 0, b: 0, a: 0 },
		}],
		depthStencilAttachment: {
			attachment: renderer.depthTexture.createView(),
			depthLoadValue: 1.0,
			depthStoreOp: "store",
			stencilLoadValue: 0,
			stencilStoreOp: "store",
		}
	};
	const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
	passEncoder.setPipeline(renderer.pipeline);

	if(sceneObject){
		passEncoder.setVertexBuffer(0, sceneObject.bufPositions);
		passEncoder.setVertexBuffer(1, sceneObject.bufColors);
		passEncoder.setBindGroup(0, renderer.uniformBindGroup);
		passEncoder.setViewport(0, 0, canvas.width, canvas.height, 0, 1);
		passEncoder.draw(sceneObject.n, 1, 0, 0);
	}

	if(testMesh){
		renderer.renderMesh(testMesh, worldViewProj, passEncoder);
	}

	passEncoder.endPass();

	renderer.device.defaultQueue.submit([commandEncoder.finish()]);

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