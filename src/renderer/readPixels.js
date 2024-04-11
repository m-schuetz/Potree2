let FORMAT_COLOR = 0;
let FORMAT_DEPTH = 1;

let cs = `

struct Uniforms {
	x : u32;
	y : u32;
	width: u32;
	height: u32;
	format: u32;
};

struct U32s {
	values :  array<u32>;
};

@binding(0), group(0) var<uniform> uniforms : Uniforms,
@binding(1), group(0) var source : texture_2d<f32>,
@binding(2), group(0) var<storage> target : @access(read_write)  U32s,

@compute
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {

	if(GlobalInvocationID.x > uniforms.width){
		return;
	}
	if(GlobalInvocationID.y > uniforms.height){
		return;
	}

	var coords : vec2<i32>;
	coords.x = i32(uniforms.x + GlobalInvocationID.x);
	coords.y = i32(uniforms.y + GlobalInvocationID.y);

	var color : vec4<f32> = textureLoad(source, coords, 0);

	var index : u32 = uniforms.width * GlobalInvocationID.y + GlobalInvocationID.x;

	if(uniforms.format == ${FORMAT_COLOR}u){
		target.values[index] = u32(color.r * 256.0);
		// TODO
	}else if(uniforms.format == ${FORMAT_DEPTH}u){
		target.values[index] = bitcast<u32>(color.r);
	}

	// target.values[index] = bitcast<u32>(color.r);
}
`;

let pipeline = null;
let ssbo = null;
let uniformBuffer = null;

function init(renderer) {
  if (pipeline !== null) {
    return;
  }

  let { device } = renderer;

  let ssboSize = 128 * 128 * 4;
  ssbo = renderer.createBuffer(ssboSize);

  pipeline = device.createComputePipeline({
    compute: {
      module: device.createShaderModule({ code: cs }),
      entryPoint: "main",
    },
  });

  let uniformBufferSize = 256;
  uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

function read(renderer, texture, x, y, width, height, callback, format) {
  init(renderer);

  let { device } = renderer;

  let bindGroup = renderer.device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: texture.createView() },
      { binding: 2, resource: { buffer: ssbo } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();

  {
    // update uniforms
    let source = new ArrayBuffer(256);
    let view = new DataView(source);

    view.setUint32(0, x, true);
    view.setUint32(4, y, true);
    view.setUint32(8, width, true);
    view.setUint32(12, height, true);
    view.setUint32(16, format, true);

    renderer.device.queue.writeBuffer(
      uniformBuffer,
      0,
      source,
      0,
      source.byteLength
    );
  }

  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatch(width, height);
  passEncoder.end();

  device.queue.submit([commandEncoder.finish()]);

  renderer.readBuffer(ssbo, 0, width * height * 4).then((result) => {
    let array = new Float32Array(result);
    let db = Math.max(...array);

    callback({ d: db });
  });
}

export function readPixels(renderer, texture, x, y, width, height, callback) {
  return read(renderer, texture, x, y, width, height, callback, FORMAT_COLOR);
}

export function readDepth(renderer, texture, x, y, width, height, callback) {
  return read(renderer, texture, x, y, width, height, callback, FORMAT_DEPTH);
}
