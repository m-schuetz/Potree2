

// import glslangModule from "../../../libs/glslang/glslang.js";

// let glslang = null;

// let csPass1 = `

// #version 450

// layout(local_size_x = 128, local_size_y = 1) in;

// layout(set = 0, binding = 0) uniform Uniforms {
// 	uint bufferSize;
// } uniforms;

// layout(std430, set = 0, binding = 1) buffer SSBO {
// 	uint objdata[];
// };

// layout(std430, set = 0, binding = 2) buffer SSBO_out {
// 	uint numLines;
// 	uint numVertices;
// 	uint numTexCoords;
// 	uint numNormals;
// 	uint numFaces;
// };

// layout(std430, set = 0, binding = 3) buffer SSBO_lines {
// 	uint lines_starts[];
// };

// uint readCharacter(uint index){
// 	uint wordIndex = index / 4;
// 	uint word = objdata[wordIndex];

// 	uint wordByteIndex = uint(mod(index, 4u));
// 	uint character = (word >> (wordByteIndex * 8)) & 0xFF;

// 	return character;
// }

// void main(){

// 	uint index = gl_GlobalInvocationID.x;

// 	if(index >= uniforms.bufferSize){
// 		return;
// 	}

// 	// uint wordIndex = index / 4;
// 	// uint word = objdata[wordIndex];

// 	// uint wordByteIndex = uint(mod(index, 4u));
// 	// uint character = (word >> (wordByteIndex * 8)) & 0xFF;

// 	uint character = readCharacter(index);

// 	// 10: newline
// 	// 13: carriage return
// 	// 102: f
// 	// 110: n
// 	// 116: t
// 	// 118: v
// 	if(character == 10){
// 		uint lineIndex = atomicAdd(numLines, 1);

// 		lines_starts[lineIndex] = index;


// 		uint tokenPos = index + 1;

// 		if(readCharacter(tokenPos) == 13){
// 			tokenPos++;
// 		}

// 		uint token = readCharacter(tokenPos);

// 		if(token == 118){
// 			// v

// 			if(readCharacter(tokenPos + 1) == 110){
// 				// vn

// 				atomicAdd(numNormals, 1);

// 			}else if(readCharacter(tokenPos + 1) == 116){
// 				// vt

// 				atomicAdd(numTexCoords, 1);
// 			}else{
// 				// assume v

// 				atomicAdd(numVertices, 1);
// 			}
// 		}else if(token == 102){
// 			atomicAdd(numFaces, 1);
// 		}


// 	}

// }

// `;


export async function loadOBJ(url, renderer){

	// glslang = await glslangModule();

	let response = await fetch(url);
	let buffer = await response.arrayBuffer();

	{
		let tStart = performance.now();

		let u8 = new Uint8Array(buffer);
		let numLines = 0;
		let numVertices = 0;
		let numTexCoords = 0;
		let numNormals = 0;
		let numFaces = 0;

		for(let i = 0; i < u8.length; i++){

			if(u8[i] === 10){
				numLines++;

				if(u8[i + 1] === 13){
					i++;
				}

				let c1 = u8[i + 1];

				if(c1 === 118){
					
					let c2 = u8[i + 2];

					if(c2 === 110){
						numNormals++;
					}else if(c2 === 116){
						numTexCoords++;
					}else{
						numVertices++;
					}

				}else if(c1 === 102){
					numFaces++;
				}
			}


		}

		let duration = performance.now() - tStart;
		console.log("numLines: ", numLines);
		console.log("numVertices: ", numVertices);
		console.log("numTexCoords: ", numTexCoords);
		console.log("numNormals: ", numNormals);
		console.log("numFaces: ", numFaces);
		console.log("duration: " + duration  + "ms");

	}


	// console.log(buffer);

	// let {device} = renderer;

	// let ssbObjDataSize = buffer.byteLength + 4 - (buffer.byteLength % 4);
	// let ssboObjdata = renderer.createBuffer(ssbObjDataSize);

	// { // INIT ssboObjdata
	// 	device.defaultQueue.writeBuffer(
	// 		ssboObjdata, 0,
	// 		buffer, 0, buffer.byteLength - ( buffer.byteLength % 4)
	// 	);

	// 	if((buffer.byteLength % 4) !== 0){
	// 		let remaining = new ArrayBuffer(4);
	// 		let view = new DataView(remaining);
			
	// 		for(let i = 0; i < (buffer.byteLength % 4); i++){
	// 			let index = (buffer.byteLength % 4) + i;

	// 			let value = new DataView(buffer).getUint8(index);
	// 			view.setUint8(i, value);
	// 		}

	// 		device.defaultQueue.writeBuffer(
	// 			ssboObjdata, buffer.byteLength - (buffer.byteLength % 4),
	// 			remaining, 0, remaining.byteLength
	// 		);

	// 	}
	// }


	// let ssboResult = renderer.createBuffer(4 * 5);
	// { // INIT ssboResult
	// 	let data = new ArrayBuffer(4 * 5);
	// 	device.defaultQueue.writeBuffer(
	// 		ssboResult, 0,
	// 		data, 0, data.byteLength
	// 	);
	// }

	// let ssboLineStarts = renderer.createBuffer(100_000_000);

	// let compiled = glslang.compileGLSL(csPass1, "compute");

	// let csModule = device.createShaderModule({
	// 	code: compiled,
	// 	source: csPass1,
	// });

	// let uniformBufferSize = 4;
	// let uniformBuffer = device.createBuffer({
	// 	size: uniformBufferSize,
	// 	usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	// });

	// let pipeline = device.createComputePipeline({
	// 	computeStage: {
	// 		module: csModule,
	// 		entryPoint: "main",
	// 	}
	// });

	// let uniformData = new Uint32Array([buffer.byteLength]);
	// device.defaultQueue.writeBuffer(
	// 	uniformBuffer, 0,
	// 	uniformData.buffer, uniformData.byteOffset, uniformData.byteLength
	// );

	// const commandEncoder = device.createCommandEncoder();
	// let passEncoder = commandEncoder.beginComputePass();

	// passEncoder.setPipeline(pipeline);

	// let tStart = performance.now();

	// let bindGroup = device.createBindGroup({
	// 	layout: pipeline.getBindGroupLayout(0),
	// 	entries: [
	// 		{binding: 0, resource: {buffer: uniformBuffer}},
	// 		{binding: 1, resource: {buffer: ssboObjdata}},
	// 		{binding: 2, resource: {buffer: ssboResult}},
	// 		{binding: 3, resource: {buffer: ssboLineStarts}},
	// 	]
	// });
	// passEncoder.setBindGroup(0, bindGroup);

	// let groups = Math.ceil(buffer.byteLength / 128);
	// passEncoder.dispatch(groups, 1, 1);

	// passEncoder.endPass();
	// device.defaultQueue.submit([commandEncoder.finish()]);

	// renderer.readBuffer(ssboResult, 0, 20).then(result => {
	// 	console.log(new Uint32Array(result));

	// 	let duration = performance.now() - tStart;
	// 	console.log(duration + "ms");
	// });

	// renderer.readBuffer(ssboLineStarts, 0, 2000).then(result => {

	// 	let lineStarts = new Uint32Array(result);
	// 	lineStarts.sort( (a, b) => a - b);

	// 	console.log("line starts: ", lineStarts);

	// 	let duration = performance.now() - tStart;
	// 	console.log(duration + "ms");
	// });

	
}