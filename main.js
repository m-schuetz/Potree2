
// based on cube example from https://github.com/cx20/webgpu-test (MIT license)

import {LASLoader} from "./LASLoader.js";
import {WebGpuRenderer} from "./WebGpuRenderer.js";

let urlPointcloud = "http://mschuetz.potree.org/lion/lion.las";

let frameCount = 0;
let lastFpsMeasure = 0;

let renderer = null;
let sceneObject = null;
let worldViewProj = mat4.create();

async function loadPointcloud(url, device){

	let loader = new LASLoader(url);
	await loader.loadHeader();

	let numPoints = loader.header.numPoints;

	// create position and color buffer
	let descriptorPos = {
		size: 12 * numPoints,
		usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
	};
	let bufPositions = renderer.device.createBuffer(descriptorPos);

	let descriptorCol = {
		size: 16 * numPoints,
		usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
	};
	let bufColors = renderer.device.createBuffer(descriptorCol);

	let sceneObject = {
		n: 0,
		bufPositions: bufPositions,
		bufColors: bufColors,
	};

	let elProgress = document.getElementById("progress");

	// this async function keeps on loading new data and updating the buffers
	let asyncLoad = async () => {
		let iterator = loader.loadBatches();
		let pointsLoaded = 0;
		for await (let batch of iterator){
			
			bufPositions.setSubData(12 * pointsLoaded, batch.positions);
			bufColors.setSubData(16 * pointsLoaded, batch.colors);
			
			pointsLoaded += batch.size;

			let progress = pointsLoaded / loader.header.numPoints;
			let strProgress = `${parseInt(progress * 100)}`;
			let msg = `loading: ${strProgress}%`;
			elProgress.innerHTML = msg;

			sceneObject.n = pointsLoaded;
		}

		elProgress.innerHTML = `loading finished`;
	};

	asyncLoad();
	
	return sceneObject;
}

async function initScene(){
	sceneObject = await loadPointcloud(urlPointcloud, renderer.device);
}


function update(timestamp){

	let {canvas} = renderer;

	{ // update worldViewProj
		let proj = mat4.create();
		let view = mat4.create();

		{ // proj
			const aspect = Math.abs(canvas.width / canvas.height);
			mat4.perspective(proj, 45, aspect, 0.1, 100.0);
		}

		{ // view
			let target = vec3.fromValues(2, 5, 0);
			let r = 5;
			let x = r * Math.sin(timestamp / 1000) + target[0];
			let y = r * Math.cos(timestamp / 1000) + target[1];
			let z = 2;

			let position = vec3.fromValues(x, y, z);
			let up = vec3.fromValues(0, 0, 1);
			mat4.lookAt(view, position, target, up);
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

	passEncoder.endPass();
	renderer.device.defaultQueue.submit([commandEncoder.finish()]);

	{// compute FPS
		frameCount++;
		let timeSinceLastFpsMeasure = (performance.now() - lastFpsMeasure) / 1000;
		if(timeSinceLastFpsMeasure > 1){
			let fps = frameCount / timeSinceLastFpsMeasure;
			console.log(`fps: ${Math.round(fps)}`);
			lastFpsMeasure = performance.now();
			frameCount = 0;
		}
	}
}

function loop(timestamp){

	update(timestamp);
	render(timestamp);

	requestAnimationFrame(loop);
}

async function run(){

	let canvas = document.getElementById("canvas");
	renderer = await WebGpuRenderer.create(canvas);

	await initScene();

	loop();

}

run();