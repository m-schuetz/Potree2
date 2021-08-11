

import {Vector3, Matrix4, Geometry} from "potree";

const shaderSource = `
[[block]] struct Uniforms {
	worldView         : mat4x4<f32>;  // 64       0
	proj              : mat4x4<f32>;  // 64      64
	screen_width      : f32;          //  4     128
	screen_height     : f32;          //  4     132
	point_size        : f32;          //  4     136
	rootVoxelSize     : f32;          //  4     140
	bbMin             : vec3<f32>;    // 16     144
	bbMax             : vec3<f32>;    // 16     160
	pad_0             : f32;
	near              : f32;          //  4     176
};

[[block]] struct NodeUniforms {
	level             : u32;
};

struct Node{
	childMask         : u32;
	childOffset       : u32;
	level             : u32;
	pad2              : u32;
};

[[block]] struct Nodes{ values : [[stride(16)]] array<Node>; };

[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;
[[binding(1), group(0)]] var<storage, read> nodes : Nodes;

[[binding(3), group(1)]] var<uniform> nodeUniforms : NodeUniforms;

struct VertexIn{
	[[location(0)]] position : vec3<f32>;
	[[location(1)]] color : vec3<f32>;
	[[builtin(vertex_index)]] vertexID : u32;
};

struct VertexOut{
	[[builtin(position)]] position : vec4<f32>;
	[[location(0)]] color : vec4<f32>;
	[[location(1)]] uv : vec2<f32>;
	[[location(2)]] lod : f32;
};

struct FragmentIn{
	[[builtin(position)]] position : vec4<f32>;
	[[location(0)]] color : vec4<f32>;
	[[location(1)]] uv : vec2<f32>;
	[[location(2)]] lod : f32;
};

struct FragmentOut{
	[[builtin(frag_depth)]] depth : f32;
	[[location(0)]] color : vec4<f32>;
};

var<private> QUAD_POS : array<vec3<f32>, 6> = array<vec3<f32>, 6>(
	vec3<f32>(-1.0, -1.0, 0.0),
	vec3<f32>( 1.0, -1.0, 0.0),
	vec3<f32>( 1.0,  1.0, 0.0),

	vec3<f32>(-1.0, -1.0, 0.0),
	vec3<f32>( 1.0,  1.0, 0.0),
	vec3<f32>(-1.0,  1.0, 0.0),
);

fn toChildIndex(npos : vec3<f32>) -> u32 {

	var index = 0u;

	if(npos.x >= 0.5){
		index = index | 1u;
	}

	if(npos.y >= 0.5){
		index = index | 2u;
	}

	if(npos.z >= 0.5){
		index = index | 4u;
	}

	return index;
}

fn getLOD(vertex : VertexIn) -> u32 {

	var cubeSize = uniforms.bbMax.x - uniforms.bbMin.x;
	var npos = (vertex.position - uniforms.bbMin) / cubeSize;

	var nodeIndex = 0u;
	var depth = 0u;

	for(var i = 0u; i < 20u; i = i + 1u){

		var node = nodes.values[nodeIndex];
		var childIndex = toChildIndex(npos);

		var hasChild = (node.childMask & (1u << childIndex)) != 0u;

		if(hasChild){

			depth = depth + 1u;

			var offsetMask = 0xFFu >> (8u - childIndex);
			var bitcount = countOneBits(offsetMask & node.childMask);

			nodeIndex = node.childOffset + bitcount;

			if(npos.x >= 0.5){
				npos.x = npos.x - 0.5;
			}
			if(npos.y >= 0.5){
				npos.y = npos.y - 0.5;
			}
			if(npos.z >= 0.5){
				npos.z = npos.z - 0.5;
			}
			npos = npos * 2.0;

		}else{
			break;
		}
	}

	return depth;
}

[[stage(vertex)]]
fn main_vertex(vertex : VertexIn) -> VertexOut {

	var abc = nodes.values[0];

	var viewPos : vec4<f32> = uniforms.worldView * vec4<f32>(vertex.position, 1.0);
	var projPos : vec4<f32> = uniforms.proj * viewPos;

	let quadVertexIndex : u32 = vertex.vertexID % 6u;
	var pos_quad : vec3<f32> = QUAD_POS[quadVertexIndex];

	var lod = getLOD(vertex);

	var pointSize = 1000.0 * uniforms.rootVoxelSize / pow(2.0, f32(lod));
	pointSize = pointSize / (-viewPos.z);
	var fx : f32 = projPos.x / projPos.w;
	fx = fx + pointSize * pos_quad.x / uniforms.screen_width;
	projPos.x = fx * projPos.w;

	var fy : f32 = projPos.y / projPos.w;
	fy = fy + pointSize * pos_quad.y / uniforms.screen_height;
	projPos.y = fy * projPos.w;

	var vout : VertexOut;
	vout.position = projPos;
	vout.color = vec4<f32>(vertex.color, 1.0);

	var cubeSize = uniforms.bbMax.x - uniforms.bbMin.x;
	var npos = (vertex.position - uniforms.bbMin) / cubeSize;


	if(lod != nodeUniforms.level){
		// discard!

		vout.position = vec4<f32>(10.0, 10.0, 10.0, 1.0);
	}

	vout.uv = pos_quad.xy;
	vout.lod = f32(lod);

	// var shouldDiscard = vertex.position.x > -1.2;
	// shouldDiscard = shouldDiscard || vertex.position.y > -0.0;
	// if(shouldDiscard){
	// 	vout.position = vec4<f32>(10.0, 10.0, 10.0, 1.0);
	// }

	return vout;
}



[[stage(fragment)]]
fn main_fragment(fragment : FragmentIn) -> FragmentOut {

	var fout : FragmentOut;
	fout.color = fragment.color;

	var d = length(fragment.uv);
	var radius = uniforms.rootVoxelSize / pow(2.0, f32(fragment.lod));

	var oldDepth = uniforms.near / fragment.position.z;
	var newDepth = oldDepth + d * d * radius * 1.0;

	fout.depth = uniforms.near / newDepth;

	// keep original depth
	// fout.depth = fragment.position.z;

	// {
	// 	var w = fragment.lod;

	// 	fout.color = vec4<f32>(w / 2.0, 0.0, 0.0, 1.0);
	// }

	return fout;
}

`;



let initialized = false;
let pipeline = null;
let geometry_boxes = null;
let uniformBuffer = null;
let nodeBuffer = null;
let nodeBufferHost = null;
let bindGroup = null;
let capacity = 10_000;

function createPipeline(renderer){

	let {device} = renderer;
	
	pipeline = device.createRenderPipeline({
		vertex: {
			module: device.createShaderModule({code: shaderSource}),
			entryPoint: "main_vertex",
			buffers: [
				{ // position
					arrayStride: 3 * 4,
					stepMode: "instance",
					attributes: [{ 
						shaderLocation: 0,
						offset: 0,
						format: "float32x3",
					}],
				},{ // color
					arrayStride: 4,
					stepMode: "instance",
					attributes: [{ 
						shaderLocation: 1,
						offset: 0,
						format: "unorm8x4",
					}],
				}
			]
		},
		fragment: {
			module: device.createShaderModule({code: shaderSource}),
			entryPoint: "main_fragment",
			targets: [{format: "bgra8unorm"}],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'back',
		},
		depthStencil: {
			depthWriteEnabled: true,
			depthCompare: 'greater',
			format: "depth32float",
		},
	});

	return pipeline;
}

function init(renderer){

	if(initialized){
		return;
	}

	{
		pipeline = createPipeline(renderer);

		let {device} = renderer;
		const uniformBufferSize = 256;

		uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		let maxNodes = 10_000;
		nodeBufferHost = new ArrayBuffer(maxNodes * 16);
		nodeBuffer = device.createBuffer({
			size: nodeBufferHost.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

		bindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
				{binding: 1, resource: {buffer: nodeBuffer}},
			],
		});
	}

	initialized = true;

}

function updateUniforms(voxelTree, drawstate){

	let {renderer, camera} = drawstate;

	let data = new ArrayBuffer(256);
	let f32 = new Float32Array(data);
	let view = new DataView(data);

	{ // transform
		let world = new Matrix4();
		let view = camera.view;
		let worldView = new Matrix4().multiplyMatrices(view, world);

		f32.set(worldView.elements, 0);
		f32.set(camera.proj.elements, 16);
	}

	{ // misc
		let size = renderer.getSize();

		view.setFloat32(128, size.width, true);
		view.setFloat32(132, size.height, true);
		view.setFloat32(136, 5.0, true);
		view.setFloat32(140, voxelTree.getRootVoxelSize(), true);

		let box = voxelTree.root.boundingBox;
		view.setFloat32(144, box.min.x, true);
		view.setFloat32(148, box.min.y, true);
		view.setFloat32(152, box.min.z, true);
		view.setFloat32(160, box.max.x, true);
		view.setFloat32(164, box.max.y, true);
		view.setFloat32(168, box.max.z, true);

		view.setFloat32(176, camera.near, true);
	}

	renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, 256);
}

function childMaskOf(node){

	let mask = 0;

	for(let i = 0; i < node.children.length; i++){
		if(node.children[i].visible){
			mask = mask | (1 << i);
		}

	}

	return mask;
}

let nodeBindGroupCache = new Map();
function getNodeBindGroup(renderer, node){

	let data = nodeBindGroupCache.get(node);

	if(!data){

		let device = renderer.device;

		let nodeUniformBuffer = device.createBuffer({
			size: 256,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		let nodeBindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(1),
			entries: [
				{binding: 3, resource: {buffer: nodeUniformBuffer}},
			],
		});

		data = {uniformBuffer: nodeUniformBuffer, bindGroup: nodeBindGroup};
		nodeBindGroupCache.set(node, data);
	}

	return data;
}

export function render(voxelTree, drawstate){

	let {renderer} = drawstate;

	init(renderer);

	updateUniforms(voxelTree, drawstate);

	let {passEncoder} = drawstate.pass;

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);


	let nodes = []; 
	voxelTree.traverseBreadthFirst( (node) => {
		if(node.voxels.length > 0){
			nodes.push(node);
		}
	});

	{

		for(let i = 0; i < nodes.length; i++){
			let node = nodes[i];

			if(node.parent !== null){
				node.parent.childOffset = Infinity;
			}
		}

		for(let i = 0; i < nodes.length; i++){
			let node = nodes[i];

			if(node.parent !== null){
				node.parent.childOffset = Math.min(node.parent.childOffset, i);
			}
		}

		let view = new DataView(nodeBufferHost);
		for(let i = 0; i < nodes.length; i++){
			let node = nodes[i];

			let mask = childMaskOf(node);
			let childOffset = node.childOffset ?? 0;

			view.setUint32(16 * i + 0, mask, true);
			view.setUint32(16 * i + 4, childOffset, true);
			view.setUint32(16 * i + 8, node.level, true);

		}

		renderer.device.queue.writeBuffer(nodeBuffer, 0, nodeBufferHost, 0, 16 * nodes.length);
	}

	for(let i = 0; i < nodes.length; i++){

		let node = nodes[i];

		if(node.voxels.length === 0){
			return;
		}

		let nodeBindGroup = getNodeBindGroup(renderer, node);
		{
			let buffer = new Uint32Array(1);
			buffer[0] = node.level;

			renderer.device.queue.writeBuffer(nodeBindGroup.uniformBuffer, 0, buffer.buffer, 0, 4);
		}

		passEncoder.setBindGroup(1, nodeBindGroup.bindGroup);
		
		let vboPosition = renderer.getGpuBuffer(node.quads.positions);
		let vboColor = renderer.getGpuBuffer(node.quads.colors);

		passEncoder.setVertexBuffer(0, vboPosition);
		passEncoder.setVertexBuffer(1, vboColor);



		let numVertices = node.quads.positions.length / 3;
		passEncoder.draw(6, numVertices, 0, 0);

		// break;
	};


};