
import {Vector3, Matrix4} from "potree";
import {Timer} from "potree";
import {generate as generatePipeline} from "./octree/pipelineGenerator.js";
import {Gradients} from "potree";

let octreeStates = new Map();
// let gradientTexture = null;
let gradientSampler = null;
let initialized = false;
let gradientTextureMap = new Map();

function init(renderer){

	if(initialized){
		return;
	}

	// let SPECTRAL = Gradients.SPECTRAL;
	// gradientTexture	= renderer.createTextureFromArray(SPECTRAL.steps.flat(), SPECTRAL.steps.length, 1);

	gradientSampler = renderer.device.createSampler({
		magFilter: 'linear',
		minFilter: 'linear',
		mipmapFilter : 'linear',
		addressModeU: "repeat",
		addressModeV: "repeat",
		maxAnisotropy: 1,
	});
}

function getGradient(){

	let gradient = Potree.settings.gradient;

	if(!gradientTextureMap.has(gradient)){
		let texture = renderer.createTextureFromArray(
			gradient.steps.flat(), gradient.steps.length, 1);

		gradientTextureMap.set(gradient, texture);
	}

	return gradientTextureMap.get(gradient);
}
 

function getOctreeState(renderer, octree, attributeName, flags = []){

	let {device} = renderer;

	let attributes = octree.loader.attributes.attributes;
	let attribute = attributes.find(a => a.name === attributeName);

	let mapping = attributeName === "rgba" ? "rgba" : "scalar";

	let key = `${attribute.name}_${attribute.numElements}_${attribute.type.name}_${mapping}_${flags.join("_")}`;

	let state = octreeStates.get(key);

	if(!state){
		let pipeline = generatePipeline(renderer, {attribute, mapping, flags});

		const uniformBuffer = device.createBuffer({
			size: 256,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		const uniformBindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
			],
		});

		let gradientTexture = getGradient(Potree.settings.gradient);

		const miscBindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(1),
			entries: [
				{binding: 0, resource: gradientSampler},
				{binding: 1, resource: gradientTexture.createView()},
			],
		});

		state = {pipeline, uniformBuffer, uniformBindGroup, miscBindGroup};

		octreeStates.set(key, state);
	}

	return state;
}

function updateUniforms(octree, octreeState, drawstate, flags){

	let {uniformBuffer} = octreeState;
	let {renderer} = drawstate;
	let isHqsDepth = flags.includes("hqs-depth");

	let data = new ArrayBuffer(256);
	let f32 = new Float32Array(data);
	let view = new DataView(data);

	let world = octree.world;
	let camView = camera.view;
	let worldView = new Matrix4().multiplyMatrices(camView, world);

	f32.set(worldView.elements, 0);
	f32.set(camera.proj.elements, 16);

	let size = renderer.getSize();

	view.setFloat32(128, size.width, true);
	view.setFloat32(132, size.height, true);
	view.setUint32(136, isHqsDepth ? 1 : 0, true);

	if(Potree.settings.dbgAttribute === "rgba"){
		view.setUint32(140, 0, true);
	}else if(Potree.settings.dbgAttribute === "elevation"){
		view.setUint32(140, 1, true);
	}

	renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, 144);
}

function renderOctree(octree, drawstate, flags){
	
	let {renderer, pass} = drawstate;
	
	let attributeName = Potree.settings.attribute;

	let octreeState = getOctreeState(renderer, octree, attributeName, flags);

	updateUniforms(octree, octreeState, drawstate, flags);

	let {pipeline, uniformBindGroup, miscBindGroup} = octreeState;

	pass.passEncoder.setPipeline(pipeline);
	pass.passEncoder.setBindGroup(0, uniformBindGroup);

	{
		let gradientTexture = getGradient(Potree.settings.gradient);

		const miscBindGroup = renderer.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(1),
			entries: [
				{binding: 0, resource: gradientSampler},
				{binding: 1, resource: gradientTexture.createView()},
			],
		});
		pass.passEncoder.setBindGroup(1, miscBindGroup);
	}

	let nodes = octree.visibleNodes;
	let i = 0;
	for(let node of nodes){

		let vboPosition = renderer.getGpuBuffer(node.geometry.buffers.find(s => s.name === "position").buffer);
		// let vboColor = renderer.getGpuBuffer(node.geometry.buffers.find(s => s.name === "rgba").buffer);
		// let vboIntensity = renderer.getGpuBuffer(node.geometry.buffers.find(s => s.name === "intensity").buffer);
		let vboAttribute = renderer.getGpuBuffer(node.geometry.buffers.find(s => s.name === attributeName).buffer);

		pass.passEncoder.setVertexBuffer(0, vboPosition);
		pass.passEncoder.setVertexBuffer(1, vboAttribute);
		// pass.passEncoder.setVertexBuffer(2, vboIntensity);

		if(octree.showBoundingBox === true){
			let box = node.boundingBox.clone().applyMatrix4(octree.world);
			let position = box.min.clone();
			position.add(box.max).multiplyScalar(0.5);
			// position.applyMatrix4(octree.world);
			let size = box.size();
			// let color = new Vector3(...SPECTRAL.get(node.level / 5));
			let color = new Vector3(255, 255, 0);
			renderer.drawBoundingBox(position, size, color);
		}

		let numElements = node.geometry.numElements;
		pass.passEncoder.draw(numElements, 1, 0, i);

		i++;
	}
}

export function render(octrees, drawstate, flags = []){

	let {renderer} = drawstate;

	init(renderer);

	Timer.timestamp(drawstate.pass.passEncoder, "octree-start");

	for(let octree of octrees){
		renderOctree(octree, drawstate, flags);
	}

	Timer.timestamp(drawstate.pass.passEncoder, "octree-end");

};