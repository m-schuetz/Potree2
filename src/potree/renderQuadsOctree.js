
import {Vector3, Matrix4} from "potree";
import {Timer} from "potree";



let vsBase = `
struct Uniforms {
	worldView : mat4x4<f32>,
	proj : mat4x4<f32>,
	screen_width : f32,
	screen_height : f32,
	hqs_flag : u32,
	colorMode : u32,
	point_size : f32,
};

struct Node {
	numPoints   : u32,
	dbg         : u32,
	min_x       : f32,
	min_y       : f32,
	min_z       : f32,
	max_x       : f32,
	max_y       : f32,
	max_z       : f32,
};

struct AttributeDescriptor{
	offset      : u32,
	numElements : u32,
	valuetype   : u32,
	range_min   : f32,
	range_max   : f32,
	clamp       : u32,
};

struct Nodes{
	values : array<Node>,
};

struct AttributeDescriptors{
	values : [[stride(24)]] array<AttributeDescriptor>,
};

struct U32s {
	values : array<u32>,
};

let TYPES_U8            =  0u;
let TYPES_U16           =  1u;
let TYPES_U32           =  2u;
let TYPES_I8            =  3u;
let TYPES_I16           =  4u;
let TYPES_I32           =  5u;
let TYPES_F32           =  6u;
let TYPES_F64           =  7u;
let TYPES_RGBA          = 50u;
let TYPES_ELEVATION     = 51u;

let CLAMP_DISABLED      =  0u;
let CLAMP_ENABLED       =  1u;

let RED      = vec4<f32>(1.0, 0.0, 0.0, 1.0);
let GREEN    = vec4<f32>(0.0, 1.0, 0.0, 1.0);
let OUTSIDE  = vec4<f32>(10.0, 10.0, 10.0, 1.0);

let PRIMITIVE_NUM_VERTICES = 6u;

@binding(0) @group(0) var<uniform> uniforms : Uniforms;
@binding(1) @group(0) var<storage, read> attributes : AttributeDescriptors;

@binding(0) @group(1) var mySampler: sampler;
@binding(1) @group(1) var myTexture: texture_2d<f32>;

@binding(0) @group(2) var<storage, read> buffer : U32s;
@binding(0) @group(3) var<storage, read> nodes : Nodes;

struct VertexInput {
	@builtin(instance_index) instanceID : u32,
	@builtin(vertex_index) vertexID : u32,
};

struct VertexOutput {
	@builtin(position) position : vec4<f32>,
	@location(0) color : vec4<f32>,
};

fn readU8(offset : u32) -> u32{
	var ipos : u32 = offset / 4u;
	var val_u32 : u32 = buffer.values[ipos];

	// var shift : u32 = 8u * (3u - (offset % 4u));
	var shift : u32 = 8u * (offset % 4u);
	var val_u8 : u32 = (val_u32 >> shift) & 0xFFu;

	return val_u8;
}

fn readU16(offset : u32) -> u32{
	
	var first = readU8(offset + 0u);
	var second = readU8(offset + 1u);

	var value = first | (second << 8u);

	return value;
}

fn readU32(offset : u32) -> u32{
	
	var d0 = readU8(offset + 0u);
	var d1 = readU8(offset + 1u);
	var d2 = readU8(offset + 2u);
	var d3 = readU8(offset + 3u);

	var value = d0
		| (d1 <<  8u)
		| (d2 << 16u)
		| (d3 << 24u);

	return value;
}

fn readF32(offset : u32) -> f32{
	
	var d0 = readU8(offset + 0u);
	var d1 = readU8(offset + 1u);
	var d2 = readU8(offset + 2u);
	var d3 = readU8(offset + 3u);

	var value_u32 = d0
		| (d1 <<  8u)
		| (d2 << 16u)
		| (d3 << 24u);

	var value_f32 = bitcast<f32>(value_u32);

	return value_f32;
}


fn scalarToColor(vertex : VertexInput, attribute : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {

	var value : f32 = 0.0;

	if(attribute.valuetype == TYPES_U8){
		var offset = node.numPoints * attribute.offset + 1u * vertex.vertexID;
		value = f32(readU8(offset));
	}else if(attribute.valuetype == TYPES_F64){
		var offset = node.numPoints * attribute.offset + 8u * vertex.vertexID;

		var b0 = readU8(offset + 0u);
		var b1 = readU8(offset + 1u);
		var b2 = readU8(offset + 2u);
		var b3 = readU8(offset + 3u);
		var b4 = readU8(offset + 4u);
		var b5 = readU8(offset + 5u);
		var b6 = readU8(offset + 6u);
		var b7 = readU8(offset + 7u);

		var exponent_f64_bin = (b7 << 4u) | (b6 >> 4u);
		var exponent_f64 = exponent_f64_bin - 1023u;

		var exponent_f32_bin = exponent_f64 + 127u;
		var mantissa_f32 = (b6 & 0x0Fu) << 19u
			| b5 << 11u
			| b4 << 3u;
		var sign = (b7 >> 7u) & 1u;
		var value_u32 = sign << 31u
			| exponent_f32_bin << 23u
			| mantissa_f32;

		var value_f32 = bitcast<f32>(value_u32);

		value = value_f32;
	}else if(attribute.valuetype == TYPES_ELEVATION){
		value = position.z;
	}else if(attribute.valuetype == TYPES_U16){
		var offset = node.numPoints * attribute.offset + 2u * vertex.vertexID;
		value = f32(readU16(offset));
	}

	var w = (value - attribute.range_min) / (attribute.range_max - attribute.range_min);

	if(attribute.clamp == CLAMP_ENABLED){
		w = clamp(w, 0.0, 1.0);
	}

	var uv : vec2<f32> = vec2<f32>(w, 0.0);
	var color = textureSampleLevel(myTexture, mySampler, uv, 0.0);

	return color;
}

fn vectorToColor(vertex : VertexInput, attribute : AttributeDescriptor, node : Node) -> vec4<f32> {

	var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);

	if(attribute.valuetype == TYPES_RGBA){

		// var offset = node.numPoints * 33u + 4u * vertex.vertexID / 3u;
		var offset = node.numPoints * attribute.offset + 4u * (vertex.vertexID / PRIMITIVE_NUM_VERTICES);
		var r = f32(readU8(offset + 0u));
		var g = f32(readU8(offset + 1u));
		var b = f32(readU8(offset + 2u));
		var a = f32(readU8(offset + 3u));

		color = vec4<f32>(r, g, b, a) / 255.0;
	}

	return color;
}


@stage(vertex)
fn main(vertex : VertexInput) -> VertexOutput {

	{ // reference all potentially unused variables, 
		// otherwise the bind group layout breaks if they're not used in the shader
		_ = mySampler;
		_ = myTexture;

		var dbg = buffer.values[0];
		var dbg1 = attributes.values[0];
	}

	var node = nodes.values[vertex.instanceID];

	var output : VertexOutput;

	var position : vec4<f32>;
	var viewPos : vec4<f32>;
	{

		var QUAD_POS : array<vec3<f32>, 6> = array<vec3<f32>, 6>(
			vec3<f32>(-1.0, -1.0, 0.0),
			vec3<f32>( 1.0, -1.0, 0.0),
			vec3<f32>( 1.0,  1.0, 0.0),

			vec3<f32>(-1.0, -1.0, 0.0),
			vec3<f32>( 1.0,  1.0, 0.0),
			vec3<f32>(-1.0,  1.0, 0.0),
		);

		{ // 3xFLOAT
			var pointID = vertex.vertexID / PRIMITIVE_NUM_VERTICES;
			var offset = 12u * pointID;
			
			position = vec4<f32>(
				readF32(offset + 0u),
				readF32(offset + 4u),
				readF32(offset + 8u),
				1.0,
			);
			
			viewPos = uniforms.worldView * position;
			var projPos = uniforms.proj * viewPos;

			{
				let quadVertexIndex : u32 = vertex.vertexID % PRIMITIVE_NUM_VERTICES;
				var pos_quad : vec3<f32> = QUAD_POS[quadVertexIndex];
				// var pos_quad : vec3<f32>;

				// if(quadVertexIndex == 0u){
				// 	pos_quad = vec3<f32>(-1.0, -1.0, 0.0);
				// }else if(quadVertexIndex == 1u){
				// 	pos_quad = vec3<f32>( 1.0, -1.0, 0.0);
				// }else if(quadVertexIndex == 2u){
				// 	pos_quad = vec3<f32>( 1.0,  1.0, 0.0);
				// }

				var point_size = uniforms.point_size;

				var fx : f32 = projPos.x / projPos.w;
				fx = fx + point_size * pos_quad.x / uniforms.screen_width;
				projPos.x = fx * projPos.w;

				var fy : f32 = projPos.y / projPos.w;
				fy = fy + point_size * pos_quad.y / uniforms.screen_height;
				projPos.y = fy * projPos.w;
			}

			output.position = projPos;
		}
		

		// { // compressed coordinates. 1xUINT32
		// 	// var offset = 4u * vertex.vertexID;
		// 	var encoded = readU32( 4u * vertex.vertexID);

		// 	var cubeSize = node.max_x - node.min_x;
		// 	var X = (encoded >> 20u) & 0x3ffu;
		// 	var Y = (encoded >> 10u) & 0x3ffu;
		// 	var Z = (encoded >>  0u) & 0x3ffu;

		// 	var x = (f32(X) / 1024.0) * cubeSize + node.min_x;
		// 	var y = (f32(Y) / 1024.0) * cubeSize + node.min_y;
		// 	var z = (f32(Z) / 1024.0) * cubeSize + node.min_z;
		// 	position = vec4<f32>(x, y, z, 1.0);

		// 	viewPos = uniforms.worldView * position;
		// 	output.position = uniforms.proj * viewPos;	
		// }

		

		// if(vertex.instanceID != 0u){
		// 	output.position = OUTSIDE;
		// }
	}

	// in the HQS depth pass, shift points 1% further away from camera
	if(uniforms.hqs_flag > 0u){
		output.position = output.position / output.position.w;

		viewPos.z = viewPos.z * 1.01;
		
		var shifted : vec4<f32> = uniforms.proj * viewPos;
		output.position.z = shifted.z / shifted.w;
	}

	{ // COLORIZE BY ATTRIBUTE DESCRIPTORS
		var attribute = attributes.values[0];
		var value : f32 = 0.0;

		var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);

		if(attribute.numElements == 1u){
			color = scalarToColor(vertex, attribute, node, position);
		}else{
			color = vectorToColor(vertex, attribute, node);
		}

		output.color = color;
	}

	// output.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);

	return output;
}
`;

const fsBase = `

struct FragmentInput {
	@location(0) color : vec4<f32>,
};

struct FragmentOutput {
	@location(0) color : vec4<f32>,
};

@stage(fragment)
fn main(fragment : FragmentInput) -> FragmentOutput {
	var output : FragmentOutput;
	output.color = fragment.color;

	return output;
}
`;



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

export function generatePipeline(renderer, args = {}){

	let {device} = renderer;
	let {flags} = args;

	let depthWrite = true;
	let blend = {
		color: {
			srcFactor: "one",
			dstFactor: "zero",
			operation: "add",
		},
		alpha: {
			srcFactor: "one",
			dstFactor: "zero",
			operation: "add",
		},
	};

	let isAdditive = flags.includes("additive_blending");
	let format = "bgra8unorm";

	// isAdditive = true;
	if(isAdditive){
		format = "rgba16float";
		depthWrite = false;

		blend = {
			color: {
				srcFactor: "one",
				dstFactor: "one",
				operation: "add",
			},
			alpha: {
				srcFactor: "one",
				dstFactor: "one",
				operation: "add",
			},
		};
	}
	
	console.groupCollapsed("compiling octree shader");
	console.log("==== VERTEX SHADER ====");
	console.log(vsBase);
	console.log("==== FRAGMENT SHADER ====");
	console.log(fsBase);
	console.groupEnd();

	const pipeline = device.createRenderPipeline({
		vertex: {
			module: device.createShaderModule({code: vsBase}),
			entryPoint: "main",
			buffers: [],
		},
		fragment: {
			module: device.createShaderModule({code: fsBase}),
			entryPoint: "main",
			targets: [{format: format, blend: blend}],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'none',
		},
		depthStencil: {
			depthWriteEnabled: depthWrite,
			depthCompare: "greater",
			format: "depth32float",
		},
	});

	return pipeline;
}

function getGradient(renderer, pipeline, gradient){

	if(!gradientTextureMap.has(gradient)){
		let texture = renderer.createTextureFromArray(
			gradient.steps.flat(), gradient.steps.length, 1);

		let bindGroup = renderer.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(1),
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

		let pointSize = Potree.settings.pointSize;
		view.setFloat32(144, pointSize, true);
		

		renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, 256);
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

		let corrector = octree.loader.metadata.encoding === "BROTLI" ? 4 : 0;
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
		pass.passEncoder.draw(6 * numElements, 1, 0, i);

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