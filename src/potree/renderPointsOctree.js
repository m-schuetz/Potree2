
import {Vector3, Matrix4} from "potree";
import {Timer} from "potree";
import {makePipeline} from "./octree/pipelineGenerator.js";
import {Gradients, SplatType} from "potree";
import {Attribute_Custom} from "./PointCloudMaterial.js";

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

	if(typeof octree.state_id === "undefined"){
		octree.state_id = ids;
		ids++;
	}

	let k_splat = `SPLAT_TYPE_${octree.material.splatType}`;
	let key = `${octree.state_id}_${flags.join("_")}_${k_splat}`;

	let state = octreeStates.get(key);

	if(state === undefined){
		// create new state

		const uniformBuffer = device.createBuffer({
			size: 512,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		let nodesBuffer = new ArrayBuffer(10_000 * 40);
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

		let state = {};
		state.uniformBuffer = uniformBuffer;
		state.nodesBuffer = nodesBuffer;
		state.nodesGpuBuffer = nodesGpuBuffer;
		state.attributesDescBuffer = attributesDescBuffer;
		state.attributesDescGpuBuffer = attributesDescGpuBuffer;
		state.colormapBuffer = colormapBuffer;
		state.colormapGpuBuffer = colormapGpuBuffer;

		octreeStates.set(key, state);

		makePipeline(renderer, {octree, state, flags});

		return null;
	}else if(state.stage === "building"){
		// just wait until its done 
		
		return null;
	}else if(state.stage === "ready"){
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
		uniformsView.setFloat32(272, performance.now() / 1000.0, true);
		uniformsView.setFloat32(276, Potree.settings.pointSize, true);
		uniformsView.setUint32(280, Potree.settings.splatType, true);
		uniformsView.setFloat32(288, octree.spacing, true);

		let isAdditive = !(octree.loader.constructor.name === "Potree2Loader");
		uniformsView.setUint32(284, isAdditive ? 1 : 0, true);

		let attributeName = Potree.settings.attribute;
		let settings = octree?.material?.attributes?.get(attributeName);
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

			let stride = 9 * 4;

			let attributeName = Potree.settings.attribute;
			let mapping = 0;
			if(!args?.settings){
				mapping = 0;
			}else if(args?.settings?.constructor.name === "Attribute_RGB"){
				mapping = 1;
			}else if(args?.settings?.constructor.name === "Attribute_Scalar"){
				mapping = 2;
			}else if(attributeName === "elevation"){
				mapping = 3;
			}else if(args?.settings?.constructor.name === "Attribute_Listing"){
				mapping = 4;
			}else if(args?.settings?.constructor.name === "Attribute_Vector"){
				mapping = 5;
			}else if(args?.settings?.constructor.name === "Attribute_Custom"){
				mapping = 6;
			}

			if(args.settings?.shaderBinding != null){
				mapping = args.settings?.shaderBinding;
			}

			if(args.settings?.mapping){
				mapping = args.settings.mapping;
			}

			attributeView.setUint32(  index * stride +  0,         args.offset, true);
			attributeView.setUint32(  index * stride +  4,         numElements, true);
			attributeView.setUint32(  index * stride +  8,           args.type, true);
			attributeView.setFloat32( index * stride + 12,           range_min, true);
			attributeView.setFloat32( index * stride + 16,           range_max, true);
			attributeView.setUint32(  index * stride + 20,               clamp, true);
			attributeView.setUint32(  index * stride + 24,            byteSize, true);
			attributeView.setUint32(  index * stride + 28,            dataType, true);
			attributeView.setUint32(  index * stride + 32,             mapping, true);
		};

		let attributes = octree.loader.attributes;
		let selectedAttribute = Potree.settings.attribute;

		let offset = 0;
		let offsets = new Map();
		for(let attribute of attributes.attributes){
			
			offsets.set(attribute.name, offset);

			offset += attribute.byteSize;
		}

		let customAttributes = [];
		for(let [name, attribute] of octree.material.attributes){
			if(attribute instanceof Attribute_Custom){
				customAttributes.push([name, attribute]);
			}
		}
		for(let i = 0; i < customAttributes.length; i++){
			let customAttribute = customAttributes[i][1];
			customAttribute.shaderBinding = 128 + i;
		}

		let i = 0;
		for(let [attributeName, settings] of octree.material.attributes){
			// let attribute = octree.material.attributes[i];
			// let settings = octree.material?.attributes?.get(attribute.name);
			let attribute = attributes.attributes.find(a => a.name === attributeName);

			if(selectedAttribute === attributeName){
				uniformsView.setUint32(268, i, true);
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
				}else if(materialValues.constructor.name === "Attribute_Custom"){
					set(i, {
						offset       : offsets.get(attributeName),
						settings,
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

	if(octree.material.splatType !== Potree.settings.splatType){
		octree.material.splatType = Potree.settings.splatType;
	}

	let octreeState = getOctreeState(renderer, octree, attributeName, flags);
	// let octreeState = getOctreeState(renderer, octree, attributeName, {...flags, splatType: Potree.settings.splatType});

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

			let bb = node.boundingBox;
			let bbWorld = octree.boundingBox;

			let childmask = 0;
			for(let j = 0; j < 8; j++){
				let visible = node.children[j]?.visible ?? false;

				if(visible){
					childmask = childmask | (1 << j);
				}
			}

			// debugger;
			let nodeSpacing = octree.spacing / (2 ** node.level);

			view.setUint32 (40 * i +  0, node.geometry.numElements, true);
			view.setUint32 (40 * i +  4, Potree.state.renderedElements + counter, true);
			view.setFloat32(40 * i +  8, bbWorld.min.x + bb.min.x, true);
			view.setFloat32(40 * i + 12, bbWorld.min.y + bb.min.y, true);
			view.setFloat32(40 * i + 16, bbWorld.min.z + bb.min.z, true);
			view.setFloat32(40 * i + 20, bbWorld.min.x + bb.max.x, true);
			view.setFloat32(40 * i + 24, bbWorld.min.y + bb.max.y, true);
			view.setFloat32(40 * i + 28, bbWorld.min.z + bb.max.z, true);
			view.setUint32 (40 * i + 32, childmask, true);
			view.setFloat32(40 * i + 36, nodeSpacing, true);

			counter += node.geometry.numElements;
		}

		renderer.device.queue.writeBuffer(
			nodesGpuBuffer, 0, 
			nodesBuffer, 0, 40 * nodes.length
		);

		pass.passEncoder.setBindGroup(3, nodesBindGroup);
	}

	let i = 0;
	for(let node of nodes){

		let numElements = node.geometry.numElements;
		if(numElements > 0){

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


			if(octree.material.splatType === SplatType.POINTS){
				pass.passEncoder.draw(numElements, 1, 0, i);
			}else if(octree.material.splatType === SplatType.QUADS){
				// 2 tris, 6 vertices
				pass.passEncoder.draw(6 * numElements, 1, 0, i);
			}else if(octree.material.splatType === SplatType.VOXELS){
				// 6 tris, 18 vertices
				pass.passEncoder.draw(18 * numElements, 1, 0, i);
			}
		}

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