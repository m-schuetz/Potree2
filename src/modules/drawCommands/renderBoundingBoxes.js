
import {Vector3, Matrix4, Geometry} from "potree";

const shaderSource = `

struct U32s {values : array<u32>};
struct F32s {values : array<f32>};

struct Uniforms {
	worldView : mat4x4<f32>,
	proj : mat4x4<f32>,
	screen_width : f32,
	screen_height : f32,
};

struct Box{
	position    : vec3<f32>,
	size        : vec3<f32>,
	color       : vec3<f32>,
};

@binding(0) @group(0) var<uniform> uniforms              : Uniforms;
@binding(1) @group(0) var<storage, read> boxPositions    : F32s;
@binding(2) @group(0) var<storage, read> boxSizes        : F32s;
@binding(3) @group(0) var<storage, read> boxColors       : U32s;

struct VertexIn{
	@builtin(vertex_index) index : u32,
};

struct VertexOut{
	@builtin(position) pos : vec4<f32>,
	@location(0) color : vec4<f32>,
};

struct FragmentIn{
	@location(0) color : vec4<f32>,
};

struct FragmentOut{
	@location(0) color : vec4<f32>,
	@location(1) id : u32,
};

fn loadBox(boxIndex : u32) -> Box {

	var box = Box();

	box.position = vec3<f32>(
		boxPositions.values[3u * boxIndex + 0u],
		boxPositions.values[3u * boxIndex + 1u],
		boxPositions.values[3u * boxIndex + 2u],
	);

	box.size = vec3<f32>(
		boxSizes.values[3u * boxIndex + 0u],
		boxSizes.values[3u * boxIndex + 1u],
		boxSizes.values[3u * boxIndex + 2u],
	);

	box.color = vec3<f32>(
		f32((boxColors.values[boxIndex] >>  0u) & 0xFFu) / 255.0,
		f32((boxColors.values[boxIndex] >>  8u) & 0xFFu) / 255.0,
		f32((boxColors.values[boxIndex] >> 16u) & 0xFFu) / 255.0,
	);

	return box;
};


@vertex
fn main_vertex(vertex : VertexIn) -> VertexOut {

	var boxIndex = vertex.index / 72u;
	var vertIndex = vertex.index % 72u;
	var lineIndex = vertIndex / 6u;
	var localIndex = vertex.index % 6u;

	var box = loadBox(boxIndex);
	
	_ = uniforms;
	_ = &boxPositions;
	_ = &boxSizes;
	_ = &boxColors;

	var LINES : array<vec3<f32>, 24> = array<vec3<f32>, 24>(
		// BOTTOM
		vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>( 1.0, -1.0, -1.0),
		vec3<f32>( 1.0, -1.0, -1.0), vec3<f32>( 1.0,  1.0, -1.0),
		vec3<f32>( 1.0,  1.0, -1.0), vec3<f32>(-1.0,  1.0, -1.0),
		vec3<f32>(-1.0,  1.0, -1.0), vec3<f32>(-1.0, -1.0, -1.0),

		// TOP
		vec3<f32>(-1.0, -1.0,  1.0), vec3<f32>( 1.0, -1.0,  1.0),
		vec3<f32>( 1.0, -1.0,  1.0), vec3<f32>( 1.0,  1.0,  1.0),
		vec3<f32>( 1.0,  1.0,  1.0), vec3<f32>(-1.0,  1.0,  1.0),
		vec3<f32>(-1.0,  1.0,  1.0), vec3<f32>(-1.0, -1.0,  1.0),

		// VERTICAL
		vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>(-1.0, -1.0,  1.0),
		vec3<f32>( 1.0, -1.0, -1.0), vec3<f32>( 1.0, -1.0,  1.0),
		vec3<f32>( 1.0,  1.0, -1.0), vec3<f32>( 1.0,  1.0,  1.0),
		vec3<f32>(-1.0,  1.0, -1.0), vec3<f32>(-1.0,  1.0,  1.0),
	);

	var start = vec4<f32>(LINES[2u * lineIndex + 0u] * box.size / 2.0 + box.position, 1.0);
	var end = vec4<f32>(LINES[2u * lineIndex + 1u] * box.size / 2.0 + box.position, 1.0);

	var position : vec4<f32>;
	if(localIndex == 0u || localIndex == 3u|| localIndex == 5u){
		position = start;
	}else{
		position = end;
	}

	var worldPos = position;
	var viewPos = uniforms.worldView * worldPos;
	var projPos = uniforms.proj * viewPos;

	var dirScreen : vec2<f32>;
	{
		var projStart = uniforms.proj * uniforms.worldView * start;
		var projEnd = uniforms.proj * uniforms.worldView * end;

		var screenStart = projStart.xy / projStart.w;
		var screenEnd = projEnd.xy / projEnd.w;

		dirScreen = normalize(screenEnd - screenStart);
	}

	{ // apply pixel offsets to the 6 vertices of the quad

		var lineWidth = 2.0;
		var pxOffset = vec2<f32>(1.0, 0.0);

		// move vertices of quad sidewards
		if(localIndex == 0u || localIndex == 1u || localIndex == 3u){
			pxOffset = vec2<f32>(dirScreen.y, -dirScreen.x);
		}else{
			pxOffset = vec2<f32>(-dirScreen.y, dirScreen.x);
		}

		// move vertices of quad outwards
		if(localIndex == 0u || localIndex == 3u || localIndex == 5u){
			pxOffset = pxOffset - dirScreen;
		}else{
			pxOffset = pxOffset + dirScreen;
		}

		var screenDimensions = vec2<f32>(uniforms.screen_width, uniforms.screen_height);
		var adjusted = projPos.xy / projPos.w + lineWidth * pxOffset / screenDimensions;
		projPos = vec4<f32>(adjusted * projPos.w, projPos.zw);
	}


	var vout : VertexOut;
	vout.pos = projPos;
	vout.color = vec4<f32>(box.color, 1.0);

	return vout;
}

@fragment
fn main_fragment(fragment : FragmentIn) -> FragmentOut {

	_ = uniforms;
	_ = &boxPositions;
	_ = &boxSizes;
	_ = &boxColors;

	var fout : FragmentOut;
	fout.color = fragment.color;
	fout.id = 0u;

	return fout;
}
`;

let initialized = false;
let pipeline = null;
let geometry_boxes = null;
let uniformBuffer = null;
let bindGroup = null;
let capacity = 10_000;

function createPipeline(renderer){

	let {device} = renderer;
	
	pipeline = device.createRenderPipeline({
		layout: "auto",
		vertex: {
			module: device.createShaderModule({code: shaderSource}),
			entryPoint: "main_vertex",
			buffers: []
		},
		fragment: {
			module: device.createShaderModule({code: shaderSource}),
			entryPoint: "main_fragment",
			targets: [
				{format: "bgra8unorm"},
				{format: "r32uint", writeMask: 0},
			],
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

	geometry_boxes = new Geometry({
		buffers: [{
			name: "position",
			buffer: new Float32Array(3 * capacity),
		},{
			name: "scale",
			buffer: new Float32Array(3 * capacity),
		},{
			name: "color",
			buffer: new Uint8Array(4 * capacity),
		}]
	});

	{
		pipeline = createPipeline(renderer);

		let {device} = renderer;
		const uniformBufferSize = 256;

		uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		let position = geometry_boxes.buffers.find(g => g.name === "position").buffer;
		let scale = geometry_boxes.buffers.find(g => g.name === "scale").buffer;
		let color = geometry_boxes.buffers.find(g => g.name === "color").buffer;
		let vboPosition = renderer.getGpuBuffer(position);
		let vboScale = renderer.getGpuBuffer(scale);
		let vboColor = renderer.getGpuBuffer(color);

		bindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
				{binding: 1, resource: {buffer: vboPosition}},
				{binding: 2, resource: {buffer: vboScale}},
				{binding: 3, resource: {buffer: vboColor}},
			],
		});
	}

	initialized = true;

}

function updateUniforms(drawstate){

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
	}

	renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, data.byteLength);
}

export function render(boxes, drawstate){

	let {renderer} = drawstate;
	let {device} = renderer;

	init(renderer);

	updateUniforms(drawstate);

	let {passEncoder} = drawstate.pass;

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	let position = geometry_boxes.buffers.find(g => g.name === "position").buffer;
	let scale = geometry_boxes.buffers.find(g => g.name === "scale").buffer;
	let color = geometry_boxes.buffers.find(g => g.name === "color").buffer;
	let vboPosition = renderer.getGpuBuffer(position);
	let vboScale = renderer.getGpuBuffer(scale);
	let vboColor = renderer.getGpuBuffer(color);
	{
		for(let i = 0; i < boxes.length; i++){
			let box = boxes[i];
			let pos = box[0];

			position[3 * i + 0] = pos.x;
			position[3 * i + 1] = pos.y;
			position[3 * i + 2] = pos.z;

			scale[3 * i + 0] = box[1].x;
			scale[3 * i + 1] = box[1].y;
			scale[3 * i + 2] = box[1].z;

			color[4 * i + 0] = box[2].x;
			color[4 * i + 1] = box[2].y;
			color[4 * i + 2] = box[2].z;
			color[4 * i + 3] = 255;
		}

		device.queue.writeBuffer(vboPosition, 0, position.buffer, 0, position.byteLength);
		device.queue.writeBuffer(vboScale, 0, scale.buffer, 0, scale.byteLength);
		device.queue.writeBuffer(vboColor, 0, color.buffer, 0, color.byteLength);
	}

	{ // wireframe
		let numBoxes = boxes.length;
		let numVertices = 72 * numBoxes;
		passEncoder.draw(numVertices, 1, 0, 0);
	}

};