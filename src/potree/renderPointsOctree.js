
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

function getGradient(renderer, pipeline, gradient){

	if(!gradientTextureMap.has(gradient)){

		let texture = renderer.createTextureFromArray(
			gradient.steps.flat(), gradient.steps.length, 1);

		let bindGroup = renderer.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(1),
			// layout: layout,
			entries: [
				{binding: 0, resource: gradientSampler},
				{binding: 1, resource: texture.createView()},
			],
		});

		gradientTextureMap.set(gradient, {texture, bindGroup});
	}

	return gradientTextureMap.get(gradient);
}
 
 let ids = 0;

function getOctreeState(renderer, octree, attributeName, flags = []){

	let {device} = renderer;


	let attributes = octree.loader.attributes.attributes;
	let mapping = "rgba";
	let attribute = attributes.find(a => a.name === mapping);

	// let key = `${attribute.name}_${attribute.numElements}_${attribute.type.name}_${mapping}_${flags.join("_")}`;

	if(typeof octree.state_id === "undefined"){
		octree.state_id = ids;
		ids++;
	}

	let key = `${octree.state_id}_${flags.join("_")}`;

	let state = octreeStates.get(key);

	if(!state){
		let pipeline = generatePipeline(renderer, {attribute, mapping, flags});

		const uniformBuffer = device.createBuffer({
			size: 256,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		// let gradientTexture = getGradient(Potree.settings.gradient);

		let nodesBuffer = new ArrayBuffer(10_000 * 32);
		let nodesGpuBuffer = device.createBuffer({
			size: nodesBuffer.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

		let attributesDescBuffer = new ArrayBuffer(1024);
		let attributesDescGpuBuffer = device.createBuffer({
			size: attributesDescBuffer.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

		let nodesBindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(3),
			entries: [
				{binding: 0, resource: {buffer: nodesGpuBuffer}},
			],
		});

		const uniformBindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
				{binding: 1, resource: {buffer: attributesDescGpuBuffer}},
			],
		});

		// const miscBindGroup = device.createBindGroup({
		// 	layout: pipeline.getBindGroupLayout(1),
		// 	entries: [
		// 		{binding: 0, resource: gradientSampler},
		// 		{binding: 1, resource: gradientTexture.createView()},
		// 	],
		// });

		state = {
			pipeline, uniformBuffer, uniformBindGroup, 
			nodesBuffer, nodesGpuBuffer, nodesBindGroup,
			attributesDescBuffer, attributesDescGpuBuffer
		};

		octreeStates.set(key, state);
	}

	return state;
}

const TYPES = {
	U8:         0,
	U16:        1,
	U32:        2,
	I8:         3,
	I16:        4,
	I32:        5,
	F32:        6,
	F64:        7,
	RGBA:      50,
	ELEVATION: 51,
};

function updateUniforms(octree, octreeState, drawstate, flags){

	{
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


	{
		let {attributesDescBuffer, attributesDescGpuBuffer} = octreeState;
		let {renderer} = drawstate;

		let view = new DataView(attributesDescBuffer);

		let selectedAttribute = Potree.settings.attribute;

		let set = (args) => {

			let clampBool = args.clamp ?? false;
			let clamp = clampBool ? 1 : 0;

			view.setUint32(   0,             args.offset, true);
			view.setUint32(   4,        args.numElements, true);
			view.setUint32(   8,               args.type, true);
			view.setFloat32( 12,           args.range[0], true);
			view.setFloat32( 16,           args.range[1], true);
			view.setUint32(  20,                   clamp, true);
		};

		let attributes = octree.loader.attributes;

		let offset = 0;
		let offsets = new Map();
		for(let attribute of attributes.attributes){
			
			offsets.set(attribute.name, offset);

			offset += attribute.byteSize;
		}

		let corrector = octree.loader.metadata?.encoding === "BROTLI" ? 4 : 0;
		let attribute = attributes.attributes.find(a => a.name === selectedAttribute);

		if(selectedAttribute === "rgba"){
			set({
				offset       : offsets.get(selectedAttribute) + corrector,
				numElements  : attribute.numElements,
				type         : TYPES.RGBA,
				range        : [0, 255],
			});
		}else if(selectedAttribute === "elevation"){
			set({
				offset       : 0,
				numElements  : 1,
				type         : TYPES.ELEVATION,
				range        : [0, 200],
				clamp        : true,
			});
		}else if(selectedAttribute === "intensity"){
			
			set({
				offset       : offsets.get(selectedAttribute) + corrector,
				numElements  : attribute.numElements,
				type         : TYPES.U16,
				range        : [0, 255],
			});
		}else if(selectedAttribute === "classification"){
			set({
				offset       : offsets.get(selectedAttribute) + corrector,
				numElements  : attribute.numElements,
				type         : TYPES.U8,
				range        : [0, 32],
			});
		}else if(selectedAttribute === "number of returns"){
			set({
				offset       : offsets.get(selectedAttribute) + corrector,
				numElements  : attribute.numElements,
				type         : TYPES.U8,
				range        : [0, 4],
			});
		}else if(selectedAttribute === "gps-time"){
			set({
				offset       : offsets.get(selectedAttribute) + corrector,
				numElements  : attribute.numElements,
				type         : TYPES.F64,
				range        : [0, 10_000],
				clamp        : false,
			});
		}

		renderer.device.queue.writeBuffer(
			attributesDescGpuBuffer, 0, 
			attributesDescBuffer, 0, 1024);
	}
}

let bufferBindGroupCache = new Map();
function getCachedBufferBindGroup(renderer, pipeline, node){

	let bindGroup = bufferBindGroupCache.get(node);

	if(bindGroup){
		return bindGroup;
	}else{
		let buffer = node.geometry.buffer;
		let gpuBuffer = renderer.getGpuBuffer(buffer);

		let bufferBindGroup = renderer.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(2),
			entries: [
				{binding: 0, resource: {buffer: gpuBuffer}}
			],
		});

		bufferBindGroupCache.set(node, bufferBindGroup);

		return bufferBindGroup;
	}

	
}

function renderOctree(octree, drawstate, flags){
	
	let {renderer, pass} = drawstate;
	
	let attributeName = Potree.settings.attribute;

	let octreeState = getOctreeState(renderer, octree, attributeName, flags);
	let nodes = octree.visibleNodes;

	updateUniforms(octree, octreeState, drawstate, flags);

	let {pipeline, uniformBindGroup} = octreeState;

	pass.passEncoder.setPipeline(pipeline);
	pass.passEncoder.setBindGroup(0, uniformBindGroup);

	{
		let {bindGroup} = getGradient(renderer, pipeline, Potree.settings.gradient);
		pass.passEncoder.setBindGroup(1, bindGroup);
	}

	{
		let {nodesBuffer, nodesGpuBuffer, nodesBindGroup} = octreeState;
		let view = new DataView(nodesBuffer);

		for(let i = 0; i < nodes.length; i++){
			let node = nodes[i];

			view.setUint32(32 * i + 0, node.geometry.numElements, true);
			view.setUint32(32 * i + 4, i, true);

			let bb = node.boundingBox;
			let bbWorld = octree.boundingBox;
			view.setFloat32(32 * i +  8, bbWorld.min.x + bb.min.x, true);
			view.setFloat32(32 * i + 12, bbWorld.min.y + bb.min.y, true);
			view.setFloat32(32 * i + 16, bbWorld.min.z + bb.min.z, true);
			view.setFloat32(32 * i + 20, bbWorld.min.x + bb.max.x, true);
			view.setFloat32(32 * i + 24, bbWorld.min.y + bb.max.y, true);
			view.setFloat32(32 * i + 28, bbWorld.min.z + bb.max.z, true);
		}

		renderer.device.queue.writeBuffer(
			nodesGpuBuffer, 0, 
			nodesBuffer, 0, 32 * nodes.length
		);

		pass.passEncoder.setBindGroup(3, nodesBindGroup);
	}

	let i = 0;
	for(let node of nodes){

		let bufferBindGroup = getCachedBufferBindGroup(renderer, pipeline, node);
		pass.passEncoder.setBindGroup(2, bufferBindGroup);
		
		if(octree.showBoundingBox === true){
			let box = node.boundingBox.clone().applyMatrix4(octree.world);
			let position = box.min.clone();
			position.add(box.max).multiplyScalar(0.5);
			let size = box.size();
			let color = new Vector3(255, 255, 0);
			renderer.drawBoundingBox(position, size, color);
		}

		let numElements = node.geometry.numElements;
		pass.passEncoder.draw(numElements, 1, 0, i);
		// Potree.state.numPoints += numElements;

		i++;
	}
}

export function render(octrees, drawstate, flags = []){

	let {renderer} = drawstate;

	init(renderer);

	Timer.timestamp(drawstate.pass.passEncoder, "octree-start");

	for(let octree of octrees){

		if(octree.visible === false){
			continue;
		}

		renderOctree(octree, drawstate, flags);
	}

	Timer.timestamp(drawstate.pass.passEncoder, "octree-end");

};