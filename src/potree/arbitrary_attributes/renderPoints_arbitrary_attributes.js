
import {Vector3, Matrix4} from "potree";
import {SPECTRAL} from "../../misc/Gradients.js";
import * as Timer from "../../renderer/Timer.js";
import {getPipeline} from "./sh_scalar.js";

let octreeStates = new Map();

function getOctreeState(renderer, node){

	let {device} = renderer;

	let state = octreeStates.get(node);

	if(!state){

		const uniformBuffer = device.createBuffer({
			size: 2 * 4 * 16 + 8,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		const uNodesBuffer = device.createBuffer({
			size: 16 * 1000,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

		const uColorAdjustmentBuffer = device.createBuffer({
			size: 256,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		state = {uniformBuffer, uNodesBuffer, uColorAdjustmentBuffer};

		octreeStates.set(node, state);

	}

	return state;
}

let cache = new Map();
let sampler = null;
let gradientTexture = null;

function getBindGroup(renderer, octree, node, attributeName, pipeline){

	let group = cache.get(node);

	if(group?.attributeName === attributeName){
		return group.handle;
	}else{
		
		let {device} = renderer;
		let octreeState = getOctreeState(renderer, octree);
		let ssboAttribute = renderer.getGpuBuffer(node.geometry.buffers.find(s => s.name === attributeName).buffer);

		if(!sampler){
			sampler = device.createSampler({
				magFilter: 'linear',
				minFilter: 'linear',
				mipmapFilter : 'linear',
				addressModeU: "repeat",
				addressModeV: "repeat",
				maxAnisotropy: 1,
			});

			gradientTexture = renderer.createTextureFromArray(SPECTRAL.steps.flat(), SPECTRAL.steps.length, 1);
		}
		
		let handle = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: octreeState.uniformBuffer}},
				{binding: 1, resource: {buffer: octreeState.uColorAdjustmentBuffer}},
				{binding: 2, resource: {buffer: octreeState.uNodesBuffer}},
				{binding: 3, resource: {buffer: ssboAttribute}},
				{binding: 10, resource: sampler},
				{binding: 11, resource: gradientTexture.createView()},
			],
		});

		let group = {attributeName, handle};

		cache.set(node, group);

		return handle;
	}



}

export function render(renderer, pass, octree, camera){

	let {device} = renderer;

	let octreeState = getOctreeState(renderer, octree);

	let nodes = octree.visibleNodes;

	{ // update uniforms
		let {uniformBuffer} = octreeState;

		{ // transform
			let world = octree.world;
			let view = camera.view;
			let worldView = new Matrix4().multiplyMatrices(view, world);

			let tmp = new Float32Array(16);

			tmp.set(worldView.elements);
			device.queue.writeBuffer(
				uniformBuffer, 0,
				tmp.buffer, tmp.byteOffset, tmp.byteLength
			);

			tmp.set(camera.proj.elements);
			device.queue.writeBuffer(
				uniformBuffer, 64,
				tmp.buffer, tmp.byteOffset, tmp.byteLength
			);
		}

		{ // screen size
			let size = renderer.getSize();
			let data = new Float32Array([size.width, size.height]);
			device.queue.writeBuffer(
				uniformBuffer,
				128,
				data.buffer,
				data.byteOffset,
				data.byteLength
			);
		}

		{ // nodes
			let buffer = new Float32Array(4 * nodes.length);
			for(let i = 0; i < nodes.length; i++){
				buffer[4 * i + 0] = 0;
				buffer[4 * i + 1] = i / 200;
				buffer[4 * i + 2] = 0;
				buffer[4 * i + 3] = 1;
			}

			device.queue.writeBuffer(
				octreeState.uNodesBuffer, 0,
				buffer.buffer, buffer.byteOffset, buffer.byteLength
			);

		}
	}

	{ // color adjust
		let buffer = new ArrayBuffer(256);
		let view = new DataView(buffer);

		let scalarMin = guiContent["scalar min"];
		let scalarMax = guiContent["scalar max"];
		let shift = -scalarMin;
		let scale = 1 / (scalarMax - scalarMin);

		let gamma = guiContent["gamma"];
		let brightness = guiContent["brightness"];
		let contrast = guiContent["contrast"];

		view.setFloat32(0, shift, true);
		view.setFloat32(4, scale, true);
		view.setFloat32(8, gamma, true);
		view.setFloat32(12, brightness, true);
		view.setFloat32(16, contrast, true);

		device.queue.writeBuffer(
			octreeState.uColorAdjustmentBuffer, 0,
			buffer, 0, buffer.byteLength
		);
	}

	let {passEncoder} = pass;
	let attributeName = guiContent.attribute;
	let pipeline = getPipeline(renderer, octree, attributeName);

	Timer.timestamp(passEncoder, "points-start");
	
	passEncoder.setPipeline(pipeline);

	let i = 0;
	for(let node of nodes){

		let bindGroup = getBindGroup(renderer, octree, node, attributeName, pipeline);
		
		passEncoder.setBindGroup(0, bindGroup);

		let vboPosition = renderer.getGpuBuffer(node.geometry.buffers.find(s => s.name === "position").buffer);

		passEncoder.setVertexBuffer(0, vboPosition);

		if(octree.showBoundingBox === true){
			let box = node.boundingBox.clone().applyMatrix4(octree.world);
			let position = box.min.clone();
			position.add(box.max).multiplyScalar(0.5);
			let size = box.size();
			// let color = new Vector3(...SPECTRAL.get(node.level / 5));
			let color = new Vector3(255, 255, 0);
			renderer.drawBoundingBox(position, size, color);
		}

		let numElements = node.geometry.numElements;
		passEncoder.draw(numElements, 1, 0, i);

		i++;
	}

	Timer.timestamp(passEncoder, "points-end");

};