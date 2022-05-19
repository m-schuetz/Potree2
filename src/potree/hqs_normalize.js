
import {SplatType, Timer} from "potree";

let shaderCode = `
	struct Uniforms {
		uTest     : u32,
		x         : f32,
		y         : f32,
		width     : f32,
		height    : f32,
		near      : f32,
		window    : i32,
	};

	@binding(0) @group(0) var<uniform> uniforms : Uniforms;
	@binding(1) @group(0) var mySampler   : sampler;
	@binding(2) @group(0) var myTexture   : texture_2d<f32>;
	@binding(3) @group(0) var tex_pointID : texture_2d<u32>;
	@binding(4) @group(0) var myDepth     : texture_depth_2d;

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

	@stage(vertex)
	fn main_vs(vertex : VertexInput) -> VertexOutput {

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

		var abc : u32 = vertex.index;
		output.position = vec4<f32>(pos[abc], 0.999, 1.0);
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

	@stage(fragment)
	fn main_fs(input : FragmentInput) -> FragmentOutput {

		_ = mySampler;
		_ = myTexture;
		_ = myDepth;
		_ = tex_pointID;


		var window = uniforms.window;
		var DEFAULT_CLOSEST = 10000000.0;
		var closest = DEFAULT_CLOSEST;
		var closestCoords = vec2<f32>(0.0, 0.0);

		for(var i : i32 = -window; i <= window; i = i + 1){
		for(var j : i32 = -window; j <= window; j = j + 1){
			var coords : vec2<i32>;
			coords.x = i32(input.fragCoord.x) + i;
			coords.y = i32(input.fragCoord.y) + j;

			// var distance = sqrt(f32(i * i + j * j));

			var d : f32 = getLinearDepthAt(input, vec2<f32>(coords));

			if(d == 0.0){
				continue;
			}

			closest = min(closest, d);

			if(closest == d){
				closestCoords = input.fragCoord.xy + vec2<f32>(f32(i), f32(j));
			}
		}
		}


		var c = vec4<f32>(0.0, 0.0, 0.0, 0.0);

		for(var i : i32 = -window; i <= window; i = i + 1){
		for(var j : i32 = -window; j <= window; j = j + 1){
			var coords : vec2<i32>;
			coords.x = i32(input.fragCoord.x) + i;
			coords.y = i32(input.fragCoord.y) + j;

			var d : f32 = getLinearDepthAt(input, vec2<f32>(coords));

			if(d <= (closest * 1.01)){
				var color = textureLoad(myTexture, coords, 0);

				// // var distance = sqrt(f32(i * i + j * j)) / f32(window);
				// var distance = f32(i + j) / f32(window + window);
				// // float distance = 2.0 * length(gl_PointCoord.xy - 0.5);
				// var w = max(0.0, 1.0 - distance);
				// w = pow(w, 10.5);

				// var a = 1.0;
				
				// if(window > 0){
				// 	a = f32(window);
				// }

				var a = 1.0;
				var distance = sqrt(f32(i * i + j * j)); // / sqrt(1.0 + 1.0);
				var w = a * exp(- (pow(distance / a, 2.0)) / 0.1);

				color = color * w;

				c = c + color;
			}

			

		}
		}


		var output : FragmentOutput;

		var coords : vec2<i32>;
		coords.x = i32(input.fragCoord.x);
		coords.y = i32(input.fragCoord.y);

		if(c.w == 0.0){
			discard;
		}
		
		c = c / c.w;

		output.color = c;
		output.depth = textureLoad(myDepth, vec2<i32>(closestCoords), 0);
		output.pointID = textureLoad(tex_pointID, vec2<i32>(closestCoords), 0).r;

		return output;
	}

	@stage(fragment)
	fn main_fs_quads(input : FragmentInput) -> FragmentOutput {

		_ = mySampler;
		_ = myTexture;
		_ = myDepth;
		_ = tex_pointID;

		var color = textureLoad(myTexture, vec2<i32>(input.fragCoord.xy), 0);

		color.x = color.x / color.w;
		color.y = color.y / color.w;
		color.z = color.z / color.w;

		// color.x = color.w;
		// color.y = color.w;
		// color.z = color.w;

		var output : FragmentOutput;

		// if(c.w == 0.0){
		// 	discard;
		// }
		
		// c = c / c.w;


		output.color = color;
		output.depth = textureLoad(myDepth, vec2<i32>(input.fragCoord.xy), 0);
		output.pointID = textureLoad(tex_pointID, vec2<i32>(input.fragCoord.xy), 0).r;

		return output;
	}
`;

// let pipeline = null;
let uniformBuffer = null;

let initialized = false;
let pipeline_points = null;
let pipeline_quads = null;

function init(renderer){

	if(initialized){
		return;
	}

	let {device, swapChainFormat} = renderer;

	let module = device.createShaderModule({code: shaderCode});

	pipeline_points = device.createRenderPipeline({
		layout: "auto",
		vertex: {
			module,
			entryPoint: "main_vs",
		},
		fragment: {
			module,
			entryPoint: "main_fs",
			targets: [
				{format: "bgra8unorm"},
				{format: "r32uint", blend: undefined}
			],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'none',
		},
		depthStencil: {
				depthWriteEnabled: true,
				depthCompare: "always",
				format: "depth32float",
		},
	});

	pipeline_quads = device.createRenderPipeline({
		layout: "auto",
		vertex: {
			module,
			entryPoint: "main_vs",
		},
		fragment: {
			module,
			entryPoint: "main_fs_quads",
			targets: [
				{format: "bgra8unorm"},
				{format: "r32uint", blend: undefined}
			],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'none',
		},
		depthStencil: {
				depthWriteEnabled: true,
				depthCompare: "always",
				format: "depth32float",
		},
	});

	let uniformBufferSize = 256;
	uniformBuffer = device.createBuffer({
		size: uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	initialized = true;
}

export function hqs_normalize(source, drawstate){

	let {renderer, camera, pass} = drawstate;
	let {passEncoder} = pass;

	init(renderer);

	Timer.timestamp(passEncoder,"dilate-start");

	let sampler = renderer.device.createSampler({
		magFilter: "linear",
		minFilter: "linear",
	});

	// TODO: possible issue: re-creating bind group every frame
	// doing that because the render target attachments may change after resize

	let pipeline = pipeline_points;
	if(Potree.settings.splatType === SplatType.POINTS){
		pipeline = pipeline_points;
	}else if(Potree.settings.splatType === SplatType.QUADS){
		pipeline = pipeline_quads;
	}

	let uniformBindGroup = renderer.device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 1, resource: sampler},
			{binding: 2, resource: source.colorAttachments[0].texture.createView()},
			{binding: 3, resource: source.colorAttachments[1].texture.createView()},
			{binding: 4, resource: source.depth.texture.createView({aspect: "depth-only"})}
		],
	});

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