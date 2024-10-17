

import {Timer} from "potree";

let shaderSource = `

	struct Uniforms {
		uTest : u32,
		x : f32,
		y : f32,
		width : f32,
		height : f32,
		near : f32,
		window  : i32,
	};

	@binding(0) @group(0) var<uniform> uniforms   : Uniforms;
	@binding(2) @group(0) var myTexture           : texture_2d<f32>;
	@binding(3) @group(0) var myDepth             : texture_depth_2d;
	@binding(4) @group(0) var tex_pointID         : texture_2d<u32>;

	struct VertexInput {
		@builtin(vertex_index) index : u32,
	};

	struct VertexOutput {
		@builtin(position) position : vec4<f32>,
		@location(0) uv : vec2<f32>,
	};

	struct FragmentInput {
		@builtin(position) fragCoord : vec4<f32>,
		@location(0) uv: vec2<f32>,
	};

	struct FragmentOutput{
		@builtin(frag_depth) depth : f32,
		@location(0) color : vec4<f32>,
		@location(1) pointID : u32,
	};

	fn toLinear(depth: f32, near: f32) -> f32{
		return near / depth;
	}

	fn getLinearDepthAt(fragment : FragmentInput, depthCoord : vec2<f32>) -> f32 {
		var currentCoord = fragment.fragCoord.xy;

		var pixelDepth = textureLoad(myDepth, vec2<i32>(depthCoord), 0);
		var diff = (currentCoord - depthCoord) / f32(uniforms.window);

		var depth = uniforms.near / pixelDepth;
		var wRadius = uniforms.near * f32(uniforms.window) * depth / f32(uniforms.width);
		var wi = diff.x * diff.x + diff.y * diff.y;
		var newDepth = depth + wi * wi * wRadius;

		if(i32(fragment.fragCoord.x) == i32(depthCoord.x) && i32(fragment.fragCoord.y) == i32(depthCoord.y)){
			newDepth = depth;
		}else{
			newDepth = depth + wi * wi * wRadius;
		}

		return newDepth;
	}

	fn getWeight(fragment : FragmentInput, depthCoord : vec2<f32>) -> f32 {
		var currentCoord = fragment.fragCoord.xy;

		var pixelDepth = textureLoad(myDepth, vec2<i32>(depthCoord), 0);
		var diff = (currentCoord - depthCoord) / f32(uniforms.window);

		var depth = uniforms.near / pixelDepth;
		var wRadius = uniforms.near * f32(uniforms.window) * depth / f32(uniforms.width);
		var wi = diff.x * diff.x + diff.y * diff.y;
		var newDepth = depth + wi * wi * wRadius;

		return wi;
	}

	//=================================================
	//=================================================

	@vertex
	fn main_vertex(vertex : VertexInput) -> VertexOutput {

		var pos : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
			vec2<f32>(0.0, 0.0),
			vec2<f32>(0.1, 0.0),
			vec2<f32>(0.1, 0.1),
			vec2<f32>(0.0, 0.0),
			vec2<f32>(0.1, 0.1),
			vec2<f32>(0.0, 0.1)
		);

		var uv : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
			vec2<f32>(0.0, 1.0),
			vec2<f32>(1.0, 1.0),
			vec2<f32>(1.0, 0.0),
			vec2<f32>(0.0, 1.0),
			vec2<f32>(1.0, 0.0),
			vec2<f32>(0.0, 0.0)
		);

		var output : VertexOutput;

		output.position = vec4<f32>(pos[vertex.index], 0.999, 1.0);
		output.uv = uv[vertex.index];

		var x : f32 = uniforms.x * 2.0 - 1.0;
		var y : f32 = uniforms.y * 2.0 - 1.0;
		var width : f32 = uniforms.width * 2.0;
		var height : f32 = uniforms.height * 2.0;

		var vi = vertex.index;
		
		if(vi == 0u || vi == 3u || vi == 5u){
			output.position.x = x;
		}else{
			output.position.x = x + width;
		}

		if(vi == 0u || vi == 1u || vi == 3u){
			output.position.y = y;
		}else{
			output.position.y = y + height;
		}

		return output;
	}

	//=================================================
	//=================================================

	@fragment
	fn main_fragment(input : FragmentInput) -> FragmentOutput {

		_ = myTexture;
		_ = myDepth;
		_ = tex_pointID;

		var window : i32 = uniforms.window;
		window = 0;
		var DEFAULT_CLOSEST = 10000000.0;
		var closest = DEFAULT_CLOSEST;
		var closestCoords = vec2<f32>(0.0, 0.0);

		for(var i : i32 = -window; i <= window; i = i + 1){
		for(var j : i32 = -window; j <= window; j = j + 1){
			var coords : vec2<i32>;
			coords.x = i32(input.fragCoord.x) + i;
			coords.y = i32(input.fragCoord.y) + j;

			var d : f32 = getLinearDepthAt(input, vec2<f32>(coords));
			closest = min(closest, d);

			if(d == 0.0) {continue;}

			if(closest == d){
				closestCoords = input.fragCoord.xy + vec2<f32>(f32(i), f32(j));
			}
		}}

		var outColor = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);

		for(var dx : i32 = -window; dx <= window; dx = dx + 1){
		for(var dy : i32 = -window; dy <= window; dy = dy + 1){
			var coords : vec2<i32>;
			coords.x = i32(input.fragCoord.x) + dx;
			coords.y = i32(input.fragCoord.y) + dy;

			var fx = f32(dx);
			var fy = f32(dy);
			var ll = fx * fx + fy * fy;
			ll = ll / 5.0;
			ll = max(abs(fx), abs(fy));

			var w = exp(-ll * 100.5f);
			w =  clamp(w, 0.01f, 1.0f);

			var depth = getLinearDepthAt(input, vec2<f32>(coords));
			var color = textureLoad(myTexture, coords, 0);

			if(depth <= closest * 1.01){
				outColor.x = outColor.x + f32(color.x) * w;
				outColor.y = outColor.y + f32(color.y) * w;
				outColor.z = outColor.z + f32(color.z) * w;
				outColor.w = outColor.w + w;
			}
		}}


		if(closest != DEFAULT_CLOSEST){

			var source = closestCoords;

			var color = textureLoad(myTexture, vec2<i32>(source), 0);
			var pixelDepth = textureLoad(myDepth, vec2<i32>(source), 0);
			var d = getLinearDepthAt(input, source);

			outColor = outColor / outColor.w;

			var output = FragmentOutput();
			output.depth = pixelDepth;
			output.color = outColor;
			output.pointID = textureLoad(tex_pointID, vec2<i32>(source), 0).r;
			
			return output;
		}else{
			discard;
		}

	}
`;

let pipeline = null;
// let uniformBindGroup = null;
let uniformBuffer = null;

function init(renderer){

	if(pipeline !== null){
		return;
	}

	let {device, swapChainFormat} = renderer;

	let module = device.createShaderModule({code: shaderSource});

	pipeline = device.createRenderPipeline({
		label: "dilate",
		layout: "auto",
		vertex: {
			module: module,
			entryPoint: "main_vertex",
		},
		fragment: {
			module: module,
			entryPoint: "main_fragment",
			targets: [
				{format: "bgra8unorm"},
				{format: "r32uint"},
			],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'none',
		},
		depthStencil: {
				depthWriteEnabled: true,
				depthCompare: "greater",
				format: "depth32float",
		},
	});

	let uniformBufferSize = 256;
	uniformBuffer = device.createBuffer({
		size: uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});
}

let uniformBindGroupCache = new Map();
function getUniformBindGroup(renderer, source){

	let data = uniformBindGroupCache.get(source);

	if(data == null || data.version < source.version){
		
		let uniformBindGroup = renderer.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
				{binding: 2, resource: source.colorAttachments[0].texture.createView()},
				{binding: 3, resource: source.depth.texture.createView({aspect: "depth-only"})},
				{binding: 4, resource: source.colorAttachments[1].texture.createView()},
			],
		});

		let data = {
			version: source.version, 
			uniformBindGroup
		};

		uniformBindGroupCache.set(source, data);

	}

	return uniformBindGroupCache.get(source).uniformBindGroup;
}

export function dilate(source, drawstate){

	let {renderer, camera, pass} = drawstate;
	let {passEncoder} = pass;

	init(renderer);

	Timer.timestamp(passEncoder,"dilate-start");

	let uniformBindGroup = getUniformBindGroup(renderer, source);

	passEncoder.setPipeline(pipeline);

	{ // update uniforms
		let source = new ArrayBuffer(32);
		let view = new DataView(source);

		let size = Potree.settings.pointSize;
		let window = Math.round((size - 1) / 2);

		view.setUint32(0, 5, true);
		view.setFloat32(4, 0, true);
		view.setFloat32(8, 0, true);
		view.setFloat32(12, 1, true);
		view.setFloat32(16, 1, true);
		view.setFloat32(20, camera.near, true);
		view.setInt32(24, window, true);
		
		renderer.device.queue.writeBuffer(
			uniformBuffer, 0,
			source, 0, source.byteLength
		);

		passEncoder.setBindGroup(0, uniformBindGroup);
	}

	passEncoder.draw(6, 1, 0, 0);

	Timer.timestamp(passEncoder,"dilate-end");

}