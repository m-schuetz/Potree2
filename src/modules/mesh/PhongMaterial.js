
import {Vector3, Matrix4} from "../../math/math.js";

const vs = `
[[block]] struct Uniforms {
	[[offset(0)]] worldView : mat4x4<f32>;
	[[offset(64)]] proj : mat4x4<f32>;
	[[offset(128)]] numPointLights : u32;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

[[location(0)]] var<in> in_position : vec4<f32>;
[[location(1)]] var<in> in_color : vec4<f32>;
[[location(2)]] var<in> in_uv : vec2<f32>;
[[location(3)]] var<in> in_normal : vec4<f32>;

[[builtin(position)]] var<out> Position : vec4<f32>;

[[location(0)]] var<out> out_color : vec4<f32>;
[[location(1)]] var<out> out_uv : vec2<f32>;
[[location(2)]] var<out> out_position : vec4<f32>;
[[location(3)]] var<out> out_normal : vec4<f32>;

[[stage(vertex)]]
fn main() -> void {

	Position = uniforms.proj * uniforms.worldView * in_position;

	out_color = in_color;
	// out_color = vec4<f32>(0.5, 0.5, 0.5, 1.0);
	out_uv = in_uv;
	out_position = uniforms.worldView * in_position;
	out_normal = in_normal;

	return;
}
`;

const fs = `

[[block]] struct PointLight {
	[[offset(0)]] position : vec4<f32>;
};

[[block]] struct PointLights {
	[[offset(0)]] values : [[stride(16)]] array<PointLight>;
};

[[block]] struct Uniforms {
	[[offset(0)]] worldView : mat4x4<f32>;
	[[offset(64)]] proj : mat4x4<f32>;
	[[offset(128)]] numPointLights : u32;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;
[[binding(1), set(0)]] var<storage_buffer> pointLights : PointLights;

[[location(0)]] var<in> in_color : vec4<f32>;
[[location(1)]] var<in> in_uv : vec2<f32>;
[[location(2)]] var<in> in_position : vec4<f32>;
[[location(3)]] var<in> in_normal : vec4<f32>;

[[location(0)]] var<out> out_color : vec4<f32>;

[[stage(fragment)]]
fn main() -> void {

	out_color = vec4<f32>(
		in_normal.x, 
		in_normal.y, 
		in_normal.z,
		1.0
	);


	{


		var lightPos : vec4<f32> = pointLights.values[0].position;
		var L : vec3<f32> = normalize(lightPos.xyz - in_position.xyz);
		var V : vec3<f32> = vec3<f32>(0, 0, 1.0);
		var H : vec3<f32> = normalize(V + L);
		var N : vec3<f32> = (uniforms.worldView * vec4<f32>(in_normal.xyz, 0.0)).xyz;

		N = normalize(N);

		var lightColor : vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);

		var diff : f32 = max(dot(N, L), 0.0);
		var diffuse : vec3<f32> = diff * lightColor;

		var shininess : f32 = 100.0;
		var spec : f32 = pow(max(dot(N, H), 0.0), shininess);
		var specular : vec3<f32> = lightColor * spec;

		out_color.r = in_color.r * diffuse.r + specular.r;
		out_color.g = in_color.g * diffuse.g + specular.g;
		out_color.b = in_color.b * diffuse.b + specular.b;

	}

	return;
}
`;
let initialized = false;
let pipeline = null;
let uniformBuffer = null;
let ssbo_pointLights = null;

function initialize(renderer){

	if(initialized){
		return;
	}

	let {device} = renderer;

	pipeline = device.createRenderPipeline({
		vertexStage: {
			module: device.createShaderModule({code: vs}),
			entryPoint: "main",
		},
		fragmentStage: {
			module: device.createShaderModule({code: fs}),
			entryPoint: "main",
		},
		primitiveTopology: "triangle-list",
		depthStencilState: {
			depthWriteEnabled: true,
			depthCompare: "less",
			format: "depth24plus-stencil8",
		},
		vertexState: {
			vertexBuffers: [
				{ // position
					arrayStride: 3 * 4,
					attributes: [{ 
						shaderLocation: 0,
						offset: 0,
						format: "float3",
					}],
				},{ // color
					arrayStride: 4 * 4,
					attributes: [{ 
						shaderLocation: 1,
						offset: 0,
						format: "float4",
					}],
				},{ // uv
					arrayStride: 2 * 4,
					attributes: [{ 
						shaderLocation: 2,
						offset: 0,
						format: "float2",
					}],
				},{ // normal
					arrayStride: 3 * 4,
					attributes: [{ 
						shaderLocation: 3,
						offset: 0,
						format: "float3",
					}],
				},
			],
		},
		rasterizationState: {
			cullMode: "none",
		},
		colorStates: [{
				format: "bgra8unorm",
		}],
	});

	const uniformBufferSize = 256; 

	uniformBuffer = device.createBuffer({
		size: uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	let maxLights = 100;
	ssbo_pointLights = renderer.createBuffer(maxLights * 16);

}

export function render(renderer, pass, node, camera, renderables){
	
	let {device} = renderer;

	initialize(renderer);

	let pointLights = renderables.get("PointLight") ?? [];
	{ // update uniforms
		let world = node.world;
		let view = camera.view;
		let worldView = new Matrix4().multiplyMatrices(view, world);

		let tmp = new Float32Array(16);

		tmp.set(worldView.elements);
		device.defaultQueue.writeBuffer(
			uniformBuffer, 0,
			tmp.buffer, tmp.byteOffset, tmp.byteLength
		);

		tmp.set(camera.proj.elements);
		device.defaultQueue.writeBuffer(
			uniformBuffer, 64,
			tmp.buffer, tmp.byteOffset, tmp.byteLength
		);

		tmp = new Uint32Array([pointLights.length]);
		device.defaultQueue.writeBuffer(
			uniformBuffer, 128,
			tmp.buffer, tmp.byteOffset, tmp.byteLength
		);
	}

	
	if(pointLights.length > 0){

		let data = new Float32Array(pointLights.length * 4);
		for(let i = 0; i < pointLights.length; i++){
			let light = pointLights[i];
			// let lightPos = light.position;
			let lightPos = light.position.clone().applyMatrix4(camera.view);

			data[4 * i + 0] = lightPos.x;
			data[4 * i + 1] = lightPos.y;
			data[4 * i + 2] = lightPos.z;
			data[4 * i + 3] = 0.0;
		}
		
		device.defaultQueue.writeBuffer(
			ssbo_pointLights, 0,
			data.buffer, 0, data.byteLength
		);
	}


	let {passEncoder} = pass;
	let vbos = renderer.getGpuBuffers(node.geometry);

	passEncoder.setPipeline(pipeline);

	let bindGroup = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 1, resource: {buffer: ssbo_pointLights}},
		]
	});

	passEncoder.setBindGroup(0, bindGroup);

	passEncoder.setVertexBuffer(0, vbos[0].vbo);
	passEncoder.setVertexBuffer(1, vbos[1].vbo);
	passEncoder.setVertexBuffer(2, vbos[2].vbo);
	passEncoder.setVertexBuffer(3, vbos[3].vbo);

	if(node.geometry.indices){
		let indexBuffer = renderer.getGpuBuffer(node.geometry.indices);

		passEncoder.setIndexBuffer(indexBuffer, "uint32", 0, indexBuffer.byteLength);

		let numIndices = node.geometry.indices.length;
		passEncoder.drawIndexed(numIndices);
	}else{
		let numElements = node.geometry.numElements;
		passEncoder.draw(numElements, 1, 0, 0);
	}

}

export class PhongMaterial{

	constructor(){

	}
	
}