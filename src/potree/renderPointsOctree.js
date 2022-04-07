
import {Vector3, Matrix4} from "potree";
import {Timer} from "potree";
import {generate as generatePipeline} from "./octree/pipelineGenerator.js";
import {Gradients} from "potree";

let octreeStates = new Map();
let gradientSampler_repeat = null;
let gradientSampler_clamp = null;
let initialized = false;
let gradientTextureMap = new Map();

function init(renderer){

	if(initialized){
		return;
	}

	gradientSampler_repeat = renderer.device.createSampler({
		label: "gradient_sampler_repeat",
		magFilter: 'linear',
		minFilter: 'linear',
		mipmapFilter : 'linear',
		addressModeU: "repeat",
		addressModeV: "repeat",
		maxAnisotropy: 1,
	});

	gradientSampler_clamp = renderer.device.createSampler({
		label: "gradient_sampler_clamp",
		magFilter: 'linear',
		minFilter: 'linear',
		mipmapFilter : 'linear',
		addressModeU: "clamp-to-edge",
		addressModeV: "clamp-to-edge",
		maxAnisotropy: 1,
	});
}

function getGradient(renderer, pipeline, gradient){

	if(!gradientTextureMap.has(gradient)){

		let texture = renderer.createTextureFromArray(
			gradient.steps.flat(), gradient.steps.length, 1);

		let bindGroup = renderer.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(1),
			entries: [
				{binding: 0, resource: gradientSampler_repeat},
				{binding: 1, resource: gradientSampler_clamp},
				{binding: 2, resource: texture.createView()},
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
	let mapping = "intensity";
	let attribute = attributes.find(a => a.name === mapping);

	if(typeof octree.state_id === "undefined"){
		octree.state_id = ids;
		ids++;
	}

	let key = `${octree.state_id}_${flags.join("_")}`;

	let state = octreeStates.get(key);
	
	// TEST: try to use this to recompile shaders at runtime
	// {
	// 	let shaderPath = `${import.meta.url}/../octree/octree.wgsl`;
	// 	fetch(shaderPath).then(async response => {
	// 		let shaderSource = await response.text();

	// 		if(shaderSource !== state.shaderSource){
	// 			console.log("changed!");
	// 		}
	// 	});
		
	// }

	if(!state){
		state = generatePipeline(renderer, {attribute, mapping, flags});

		octreeStates.set(key, state);

		return null;
	}else if(state?.stage == "created pipeline"){
		
		let pipeline = state.pipeline;

		const uniformBuffer = device.createBuffer({
			size: 512,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

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

		let colormapBuffer = new ArrayBuffer(4 * 256);
		let colormapGpuBuffer = device.createBuffer({
			size: colormapBuffer.byteLength,
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
				{binding: 2, resource: {buffer: colormapGpuBuffer}},
			],
		});

		state = {
			pipeline, uniformBuffer, uniformBindGroup, 
			nodesBuffer, nodesGpuBuffer, nodesBindGroup,
			attributesDescBuffer, attributesDescGpuBuffer,
			colormapBuffer, colormapGpuBuffer,
			shaderSource: state.shaderSource,
			stage: "finished",
		};

		octreeStates.set(key, state);

		return state;
		
	}else if(state?.stage != "finished"){
		state.next().then(result => {
			console.log(result);

			if(result.value?.stage === "created pipeline"){

				octreeStates.set(key, result.value);

			}
		});

		return null;
	}else{
		return state;
	}


}

const TYPES = {
	DOUBLE:     0,
	FLOAT:      1,
	INT8:       2,
	UINT8:      3,
	INT16:      4,
	UINT16:     5,
	INT32:      6,
	UINT32:     7,
	INT64:      8,
	UINT64:     9,
	RGBA:      50,
	ELEVATION: 51,
};

function updateUniforms(octree, octreeState, drawstate, flags){

	let {uniformBuffer} = octreeState;
	let data = new ArrayBuffer(512);
	let uniformsView = new DataView(data);

	{
		let {renderer} = drawstate;
		let isHqsDepth = flags.includes("hqs-depth");
		
		let f32 = new Float32Array(data);
		
		let world = octree.world;
		let camView = camera.view;
		let worldView = new Matrix4().multiplyMatrices(camView, world);

		f32.set(world.elements, 0);
		f32.set(camView.elements, 16);
		f32.set(camera.proj.elements, 32);
		f32.set(worldView.elements, 48);

		let size = renderer.getSize();

		uniformsView.setFloat32(256, size.width, true);
		uniformsView.setFloat32(260, size.height, true);
		uniformsView.setUint32(264, isHqsDepth ? 1 : 0, true);

		let attributeName = Potree.settings.attribute;
		let settings = octree?.material?.attributes?.get(attributeName);

		if(!settings){
			uniformsView.setUint32(268, 0, true);
		}else if(settings?.constructor.name === "Attribute_RGB"){
			uniformsView.setUint32(268, 1, true);
		}else if(settings?.constructor.name === "Attribute_Scalar"){
			uniformsView.setUint32(268, 2, true);
		}else if(attributeName === "elevation"){
			uniformsView.setUint32(268, 3, true);
		}else if(settings?.constructor.name === "Attribute_Listing"){
			uniformsView.setUint32(268, 4, true);
		}
	}


	{
		let {attributesDescBuffer, attributesDescGpuBuffer} = octreeState;
		let {renderer} = drawstate;

		let attributeView = new DataView(attributesDescBuffer);

		let set = (index, args) => {

			let clampBool = args?.settings?.clamp ?? false;
			let clamp = clampBool ? 1 : 0;

			let byteSize = args.attribute?.byteSize ?? 0;
			let dataType = args.attribute?.type?.ordinal ?? 0;
			let numElements = args.attribute?.numElements ?? 1;

			let range_min = 0;
			let range_max = 1;

			if(args?.settings?.range){
				range_min = args.settings.range[0];
				range_max = args.settings.range[1];
			}

			let stride = 8 * 4;

			attributeView.setUint32(  index * stride +  0,         args.offset, true);
			attributeView.setUint32(  index * stride +  4,         numElements, true);
			attributeView.setUint32(  index * stride +  8,           args.type, true);
			attributeView.setFloat32( index * stride + 12,           range_min, true);
			attributeView.setFloat32( index * stride + 16,           range_max, true);
			attributeView.setUint32(  index * stride + 20,               clamp, true);
			attributeView.setUint32(  index * stride + 24,            byteSize, true);
			attributeView.setUint32(  index * stride + 28,            dataType, true);
		};

		let attributes = octree.loader.attributes;
		let selectedAttribute = Potree.settings.attribute;

		let offset = 0;
		let offsets = new Map();
		for(let attribute of attributes.attributes){
			
			offsets.set(attribute.name, offset);

			offset += attribute.byteSize;
		}

		let i = 0;
		for(let [attributeName, settings] of octree.material.attributes){
			// let attribute = octree.material.attributes[i];
			// let settings = octree.material?.attributes?.get(attribute.name);
			let attribute = attributes.attributes.find(a => a.name === attributeName);

			if(selectedAttribute === attributeName){
				uniformsView.setUint32(272, i, true);
			}

			if(attributeName === "rgba"){
				set(i, {
					offset       : offsets.get(attributeName),
					type         : TYPES.RGBA,
					range        : [0, 255],
					attribute, settings,
				});
			}
			else if(attributeName === "elevation"){
				let materialValues = octree.material.attributes.get(attributeName);
				set(i, {
					offset       : 0,
					type         : TYPES.ELEVATION,
					range        : materialValues.range,
					attribute, settings,
				});
			}
			else if(attributeName === "intensity"){
				
				set(i, {
					offset       : offsets.get(attributeName),
					type         : TYPES.UINT16,
					range        : [0, 255],
					attribute, settings,
				});
			}
			else if(attributeName === "number of returns"){
				set(i, {
					offset       : offsets.get(attributeName),
					type         : TYPES.UINT8,
					range        : [0, 4],
					attribute, settings,
				});
			}
			else if(octree.material?.attributes.has(attributeName)){
				let materialValues = octree.material.attributes.get(attributeName);

				if(materialValues.constructor.name === "Attribute_RGB"){
					set(i, {
						offset       : offsets.get(attributeName),
						type         : attribute.type.ordinal,
						range        : materialValues.range,
						attribute, settings,
					});
				}else if(materialValues.constructor.name === "Attribute_Scalar"){
					set(i, {
						offset       : offsets.get(attributeName),
						type         : attribute.type.ordinal,
						range        : materialValues.range,
						attribute, settings,
					});
				}else if(materialValues.constructor.name === "Attribute_Listing"){
					set(i, {
						offset       : offsets.get(attributeName),
						type         : attribute.type.ordinal,
						attribute, settings,
					});
				}else{
					debugger;
				}

			}
			else{
				set(i, {
					offset       : offsets.get(attributeName),
					type         : TYPES.U8,
					range        : [0, 10_000],
					attribute, settings,
				});
			}

			i++;
		}

		renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, 512);
		
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

async function renderOctree(octree, drawstate, flags){
	
	let {renderer, pass} = drawstate;
	
	let attributeName = Potree.settings.attribute;

	let octreeState = getOctreeState(renderer, octree, attributeName, flags);

	if(!octreeState){
		return;
	}

	let nodes = octree.visibleNodes;

	updateUniforms(octree, octreeState, drawstate, flags);

	{ // UPDATE COLORMAP BUFFER
		let attributeName = Potree.settings.attribute;

		let settings = octree?.material?.attributes?.get(attributeName);

		if(settings?.constructor?.name === "Attribute_Listing"){
			let {colormapBuffer, colormapGpuBuffer} = octreeState;

			let u8 = new Uint8Array(colormapBuffer);
			let defaultValue = settings.listing?.DEFAULT ?? 0;
			for(let i = 0; i < 256; i++){
				u8[4 * i + 0] = 255 * defaultValue.color[0];
				u8[4 * i + 1] = 255 * defaultValue.color[1];
				u8[4 * i + 2] = 255 * defaultValue.color[2];
				u8[4 * i + 3] = 255 * defaultValue.color[3];
			}

			for(let index of Object.keys(settings.listing)){
				if(index === "DEFAULT"){
					continue;
				}

				let value = settings.listing[index];

				u8[4 * index + 0] = 255 * value.color[0];
				u8[4 * index + 1] = 255 * value.color[1];
				u8[4 * index + 2] = 255 * value.color[2];
				u8[4 * index + 3] = 255 * value.color[3];
			}

			renderer.device.queue.writeBuffer(
				colormapGpuBuffer, 0, 
				colormapBuffer, 0, colormapBuffer.byteLength
			);
		}		
	}

	let {pipeline, uniformBindGroup} = octreeState;

	pass.passEncoder.setPipeline(pipeline);
	pass.passEncoder.setBindGroup(0, uniformBindGroup);

	{
		let {bindGroup} = getGradient(renderer, pipeline, Potree.settings.gradient);
		pass.passEncoder.setBindGroup(1, bindGroup);
	}

	{ // UPDATE NODES BUFFER
		let {nodesBuffer, nodesGpuBuffer, nodesBindGroup} = octreeState;
		let view = new DataView(nodesBuffer);

		let counter = 0;
		for(let i = 0; i < nodes.length; i++){
			let node = nodes[i];

			view.setUint32(32 * i + 0, node.geometry.numElements, true);
			view.setUint32(32 * i + 4, Potree.state.renderedElements + counter, true);
			counter += node.geometry.numElements;

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
		Potree.state.renderedElements += numElements;
		Potree.state.renderedObjects.push({node, numElements});

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