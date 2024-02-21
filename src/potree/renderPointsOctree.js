
import {Vector3, Matrix4} from "potree";
import {Timer} from "potree";
import {makePipeline} from "./octree/pipelineGenerator.js";
import {Gradients, SplatType} from "potree";
import {PMath} from "potree";

let octreeStates = new Map();
let gradientSampler_repeat = null;
let gradientSampler_clamp = null;
let initialized = false;
let gradientTextureMap = new Map();

const WGSL_NODE_BYTESIZE = 48;
const tmp = new ArrayBuffer(10_000_000);
let dbgUploadedInFrame = 0;

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

	let key_mappings = octree.material.mappings.map(m => m.name + "_" + m.inputs.join("_"))
	let key = `${octree.state_id}_${flags.join(";")}_${key_mappings}`;

	let state = octreeStates.get(key);

	if(state === undefined){
		// create new state

		octreeStates.set(key, {stage: "building"});

		// const uniformBuffer = device.createBuffer({
		// 	size: 512,
		// 	usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		// });

		const uniformBuffer = renderer.createBuffer({
			size: 512,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		let nodesBuffer = new ArrayBuffer(10_000 * WGSL_NODE_BYTESIZE);
		let nodesGpuBuffer = renderer.createBuffer({
			size: nodesBuffer.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

		let attributesDescBuffer = new ArrayBuffer(16_384);
		let attributesDescGpuBuffer = renderer.createBuffer({
			size: attributesDescBuffer.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

		let colormapBuffer = new ArrayBuffer(4 * 256);
		let colormapGpuBuffer = renderer.createBuffer({
			size: colormapBuffer.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

		// let nodesBuffer = new ArrayBuffer(10_000 * WGSL_NODE_BYTESIZE);
		// let nodesGpuBuffer = device.createBuffer({
		// 	size: nodesBuffer.byteLength,
		// 	usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		// });

		// let attributesDescBuffer = new ArrayBuffer(16_384);
		// let attributesDescGpuBuffer = device.createBuffer({
		// 	size: attributesDescBuffer.byteLength,
		// 	usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		// });

		// let colormapBuffer = new ArrayBuffer(4 * 256);
		// let colormapGpuBuffer = device.createBuffer({
		// 	size: colormapBuffer.byteLength,
		// 	usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		// });

		let state = {};
		state.uniformBuffer = uniformBuffer;
		state.nodesBuffer = nodesBuffer;
		state.nodesGpuBuffer = nodesGpuBuffer;
		state.attributesDescBuffer = attributesDescBuffer;
		state.attributesDescGpuBuffer = attributesDescGpuBuffer;
		state.colormapBuffer = colormapBuffer;
		state.colormapGpuBuffer = colormapGpuBuffer;

		octreeStates.set(key, state);

		makePipeline(renderer, {octree, state, flags, key});

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
	let data = new ArrayBuffer(1024);
	let uniformsView = new DataView(data);

	{ // UPDATE UNIFORM BUFFER
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
		// uniformsView.setUint32(280, 1, true);
		uniformsView.setUint32(280, Potree.settings.splatType, true);
		uniformsView.setFloat32(288, octree.spacing, true);

		let isAdditive = !(octree.loader.constructor.name === "Potree3Loader");
		uniformsView.setUint32(284, isAdditive ? 1 : 0, true);

		let bb = octree.getBoundingBoxWorld();
		uniformsView.setFloat32(304, bb.min.x, true);
		uniformsView.setFloat32(308, bb.min.y, true);
		uniformsView.setFloat32(312, bb.min.z, true);
		uniformsView.setFloat32(320, bb.max.z, true);
		uniformsView.setFloat32(324, bb.max.z, true);
		uniformsView.setFloat32(328, bb.max.z, true);

		// let attributeName = Potree.settings.attribute;
		// let settings = octree?.material?.attributes?.get(attributeName);
	}


	{ // UPDATE ATTRIBUTES BUFFER
		let {attributesDescBuffer, attributesDescGpuBuffer} = octreeState;
		let {renderer} = drawstate;

		let attributeView = new DataView(attributesDescBuffer);

		let set = (index, args) => {

			let clampBool = args?.settings?.clamp ?? false;
			let clamp = clampBool ? 1 : 0;

			let byteSize = args.attribute?.byteSize ?? 0;
			let dataType = args.attribute?.type?.ordinal ?? 0;
			let numElements = args.attribute?.numElements ?? 1;

			let range_min = [0, 0, 0, 0];
			let range_max = [1, 1, 1, 1];

			if(args?.settings?.range){
				if(args?.settings?.range[0] instanceof Array){
					for(let i = 0; i < args.settings.range.length; i++){
						range_min[i] = args.settings.range[0][i];
						range_max[i] = args.settings.range[1][i];
					}
				}else{
					range_min[0] = args.settings.range[0];
					range_max[0] = args.settings.range[1];
				}
				
			}

			// let attributeName = Potree.settings.attribute;
			let attributeName = args.settings.name;
			let material = octree.material;
			let mapping = material.selectedMappings.get(attributeName);

			let stride = 64;
			attributeView.setUint32(  index * stride +  0,         args.offset, true);
			attributeView.setUint32(  index * stride +  4,         numElements, true);
			attributeView.setUint32(  index * stride +  8,           args.type, true);
			attributeView.setUint32(  index * stride + 12,               clamp, true);
			attributeView.setUint32(  index * stride + 16,            byteSize, true);
			attributeView.setUint32(  index * stride + 20,            dataType, true);
			attributeView.setUint32(  index * stride + 24,       mapping.index, true);
			attributeView.setFloat32( index * stride + 32,           range_min[0], true);
			attributeView.setFloat32( index * stride + 36,           range_min[1], true);
			attributeView.setFloat32( index * stride + 40,           range_min[2], true);
			attributeView.setFloat32( index * stride + 44,           range_min[3], true);
			attributeView.setFloat32( index * stride + 48,           range_max[0], true);
			attributeView.setFloat32( index * stride + 52,           range_max[1], true);
			attributeView.setFloat32( index * stride + 56,           range_max[2], true);
			attributeView.setFloat32( index * stride + 60,           range_max[3], true);
		};

		let attributes = octree.loader.attributes;
		let selectedAttribute = Potree.settings.attribute;

		let offset = 0;
		let offsets = new Map();
		for(let attribute of attributes.attributes)
		{
			offsets.set(attribute.name, offset);
			offset += attribute.byteSize;
		}

		let i = 0;
		for(let [attributeName, settings] of octree.material.attributes){

			let attribute = attributes.attributes.find(a => a.name === attributeName);

			if(selectedAttribute === attributeName){
				uniformsView.setUint32(268, i, true);
			}

			set(i, {
				offset       : offsets.get(attributeName),
				type         : TYPES.U8,
				range        : [0, 10_000],
				attribute, settings,
			});
			
			i++;
		}

		renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, 512);
		
		renderer.device.queue.writeBuffer(
			attributesDescGpuBuffer, 0, 
			attributesDescBuffer, 0, 16_384);
	}
}

function updateNodesBuffer(octree, nodes, prefixSum, octreeState, drawstate, flags, pass){
	let {nodesBuffer, nodesGpuBuffer} = octreeState;
	let view = new DataView(nodesBuffer);

	let counter = 0;
	for(let [i, node] of nodes){
		// let node = nodes[i];

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

		let splatType = 0;
		if(node.isSmallNode){
			splatType = 0;
		}else{
			splatType = 1;
		}

		// DEBUG
		// if(node.nodeType === 0){
		// 	nodeSpacing = 123;
		// }

		view.setUint32 (WGSL_NODE_BYTESIZE * i +  0, node.geometry.numElements, true);
		view.setUint32 (WGSL_NODE_BYTESIZE * i +  4, prefixSum[i], true);
		view.setFloat32(WGSL_NODE_BYTESIZE * i +  8, bbWorld.min.x + bb.min.x, true);
		view.setFloat32(WGSL_NODE_BYTESIZE * i + 12, bbWorld.min.y + bb.min.y, true);
		view.setFloat32(WGSL_NODE_BYTESIZE * i + 16, bbWorld.min.z + bb.min.z, true);
		view.setFloat32(WGSL_NODE_BYTESIZE * i + 20, bbWorld.min.x + bb.max.x, true);
		view.setFloat32(WGSL_NODE_BYTESIZE * i + 24, bbWorld.min.y + bb.max.y, true);
		view.setFloat32(WGSL_NODE_BYTESIZE * i + 28, bbWorld.min.z + bb.max.z, true);
		view.setUint32 (WGSL_NODE_BYTESIZE * i + 32, childmask, true);
		view.setFloat32(WGSL_NODE_BYTESIZE * i + 36, nodeSpacing, true);
		view.setUint32 (WGSL_NODE_BYTESIZE * i + 40, splatType, true);
		// view.setUint32 (WGSL_NODE_BYTESIZE * i + 44, node.gpuChunks[0].offset, true);

		counter += node.geometry.numElements;
	}

	renderer.device.queue.writeBuffer(
		nodesGpuBuffer, 0, 
		nodesBuffer, 0, WGSL_NODE_BYTESIZE * octree.visibleNodes.length
	);
}

let bufferBindGroupCache = new Map();
// let dbgBuffer = null;
function getCachedBufferBindGroup(renderer, pipeline, node){

	let bindGroup = bufferBindGroupCache.get(node);

	// Remove outdated bind groups. (Old buffer was destroyed, a new buffer was later reloaded)
	if(bindGroup && bindGroup.geometry_id !== node.geometry.id){
		// console.log(`removing old bind group for ${node.name}`);
		bufferBindGroupCache.delete(bindGroup);
		bindGroup = null;
	}

	if(bindGroup){
		return bindGroup;
	}else{
		let buffer = node.geometry.buffer;
		let gpuBuffer = renderer.getGpuBuffer(buffer);
		gpuBuffer.label = node.name;

		dbgUploadedInFrame += buffer.byteLength;

		let bufferBindGroup = renderer.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(2),
			entries: [
				{binding: 0, resource: {buffer: gpuBuffer}}
			],
		});
		bufferBindGroup.geometry_id = node.geometry.id;

		bufferBindGroupCache.set(node, bufferBindGroup);

		return bufferBindGroup;
	}

	
}

async function renderOctree(octree, drawstate, flags){

	let tStart = performance.now();
	
	let {renderer, pass} = drawstate;
	
	let attributeName = Potree.settings.attribute;

	let octreeState_points = getOctreeState(renderer, octree, attributeName, [...flags, "SPLAT_TYPE_0"]);
	let octreeState_quads = getOctreeState(renderer, octree, attributeName, [...flags, "SPLAT_TYPE_1"]);

	if(!octreeState_points) return;
	if(!octreeState_quads) return;

	octreeState_points.pipelinePromise.then(pipeline => {
		octreeState_points.pipeline = pipeline;
	});
	octreeState_quads.pipelinePromise.then(pipeline => {
		octreeState_quads.pipeline = pipeline;
	});

	if(!octreeState_points.pipeline) return;
	if(!octreeState_quads.pipeline) return;

	let nodes = octree.visibleNodes;
	nodes = nodes.filter(node => node.numPoints != null ? node.numPoints > 0 : true);

	updateUniforms(octree, octreeState_quads, drawstate, flags);
	updateUniforms(octree, octreeState_points, drawstate, flags);

	let mapping = octree.material.selectedMappings.get(attributeName);

	if(mapping)
	{ // UPDATE COLORMAP BUFFER
		let attributeName = Potree.settings.attribute;
		let listing = mapping.listing;

		for(let state of [octreeState_quads, octreeState_points]){
			if(listing){
				let {colormapBuffer, colormapGpuBuffer} = state;

				let u8 = new Uint8Array(colormapBuffer);
				let defaultValue = listing?.DEFAULT ?? 0;
				for(let i = 0; i < 256; i++){
					u8[4 * i + 0] = 255 * defaultValue.color[0];
					u8[4 * i + 1] = 255 * defaultValue.color[1];
					u8[4 * i + 2] = 255 * defaultValue.color[2];
					u8[4 * i + 3] = 255 * defaultValue.color[3];
				}

				for(let index of Object.keys(listing)){
					if(index === "DEFAULT"){
						continue;
					}

					let value = listing[index];

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
	}

	let largeNodes = [];
	let smallNodes = [];
	let prefixSum = [];
	let count = 0;

	for(let i = 0; i < nodes.length; i++){
		let node = nodes[i];

		if(node.__pixelSize <= 500 || true){
			smallNodes.push([i, node]);
			node.isSmallNode = true;
		}else{
			largeNodes.push([i, node]);
			node.isSmallNode = false;
		}

		let numElements = node.geometry.numElements;
		Potree.state.renderedElements += numElements;
		Potree.state.renderedObjects.push({node, numElements});

		prefixSum.push(count);
		count += numElements;
	}

	updateNodesBuffer(octree, largeNodes, prefixSum, octreeState_quads, drawstate, flags, pass);
	updateNodesBuffer(octree, smallNodes, prefixSum, octreeState_points, drawstate, flags, pass);

	// { // DRAW LARGE NODES AS QUADS
	// 	let {pipeline, uniformBindGroup, nodesBindGroup} = octreeState_quads;
	// 	let {bindGroup} = getGradient(renderer, pipeline, Potree.settings.gradient);

	// 	pass.passEncoder.setPipeline(pipeline);
	// 	pass.passEncoder.setBindGroup(0, uniformBindGroup);
	// 	pass.passEncoder.setBindGroup(1, bindGroup);
	// 	pass.passEncoder.setBindGroup(3, nodesBindGroup);

	// 	for(let [index, node] of largeNodes){

	// 		let numElements = node.geometry.numElements;

	// 		let bufferBindGroup = getCachedBufferBindGroup(renderer, pipeline, node);
	// 		pass.passEncoder.setBindGroup(2, bufferBindGroup);

	// 		if(node.dirty){

	// 			let gpuBuffer = renderer.getGpuBuffer(node.geometry.buffer);
	// 			renderer.device.queue.writeBuffer(
	// 				gpuBuffer, 0, node.geometry.buffer, 0, node.geometry.buffer.byteLength);

	// 			node.dirty = false;
	// 		}

	// 		if(octree.showBoundingBox === true){
	// 			let box = node.boundingBox.clone().applyMatrix4(octree.world);
	// 			let position = box.min.clone();
	// 			position.add(box.max).multiplyScalar(0.5);
	// 			let size = box.size();
	// 			let color = new Vector3(255, 255, 0);
	// 			renderer.drawBoundingBox(position, size, color);
	// 		}

			
	// 		pass.passEncoder.draw(6 * numElements, 1, 0, index);
	// 	}
	// }

	
	{ // DRAW SMALL NODES AS POINTS

		let {pipeline, uniformBindGroup, nodesBindGroup} = octreeState_points;
		let {bindGroup} = getGradient(renderer, pipeline, Potree.settings.gradient);

		pass.passEncoder.setPipeline(pipeline);
		pass.passEncoder.setBindGroup(0, uniformBindGroup);
		pass.passEncoder.setBindGroup(1, bindGroup);
		pass.passEncoder.setBindGroup(3, nodesBindGroup);

		for(let [index, node] of smallNodes){

			let numElements = node.geometry.numElements;

			let bufferBindGroup = getCachedBufferBindGroup(renderer, pipeline, node);
			pass.passEncoder.setBindGroup(2, bufferBindGroup);

			if(node.dirty){
				let gpuBuffer = renderer.getGpuBuffer(node.geometry.buffer);
				renderer.device.queue.writeBuffer(
					gpuBuffer, 0, node.geometry.buffer, 0, node.geometry.buffer.byteLength);

				node.dirty = false;
			}

			if(octree.showBoundingBox === true){
				let box = node.boundingBox.clone().applyMatrix4(octree.world);
				let position = box.min.clone();
				position.add(box.max).multiplyScalar(0.5);
				let size = box.size();
				let color = new Vector3(255, 255, 0);
				renderer.drawBoundingBox(position, size, color);
			}

			// let allowed = [
			// 	// "r046163",
			// 	// "r046167",
			// 	// "r402621",
			// 	// "r402230",
			// 	// "r402231",
			// 	// "r402232",
			// 	"r402233",
			// ];

			// if(node.name.length > 6)
			// if(node.name === "r046341")
			// if(allowed.includes(node.name))
			{
				pass.passEncoder.draw(1 * numElements, 1, 0, index);
				// console.log(node.name);
			}
		}
	}
	
}

export function render(octrees, drawstate, flags = []){

	dbgUploadedInFrame = 0;

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