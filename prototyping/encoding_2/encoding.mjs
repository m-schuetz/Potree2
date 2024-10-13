
import {promises as fsp} from "fs";
import JSON5 from './json5.mjs';


let path = "g:/temp/retz/octree.bin";
let file = await fsp.open(path);

let byteOffset = 2806623798;
let byteSize = 219336;
let numVoxels = 18278;

let buffer = Buffer.alloc(byteSize);
file.read(buffer, 0, byteSize, byteOffset);

let buffer_interleaved_u8 = Buffer.alloc(3 * numVoxels);
for(let i = 0; i < numVoxels; i++){
	let r = buffer.readUint8(3 * numVoxels + 3 * i + 0);
	let g = buffer.readUint8(3 * numVoxels + 3 * i + 1);
	let b = buffer.readUint8(3 * numVoxels + 3 * i + 2);

	buffer_interleaved_u8.writeUInt8(r, 3 * i + 0);
	buffer_interleaved_u8.writeUInt8(g, 3 * i + 1);
	buffer_interleaved_u8.writeUInt8(b, 3 * i + 2);
}

let buffer_soa_u8 = Buffer.alloc(3 * numVoxels);
for(let i = 0; i < numVoxels; i++){
	let r = buffer.readUint8(3 * numVoxels + 3 * i + 0);
	let g = buffer.readUint8(3 * numVoxels + 3 * i + 1);
	let b = buffer.readUint8(3 * numVoxels + 3 * i + 2);

	buffer_soa_u8.writeUInt8(r, 0 * numVoxels + i);
	buffer_soa_u8.writeUInt8(g, 1 * numVoxels + i);
	buffer_soa_u8.writeUInt8(b, 2 * numVoxels + i);
}

let buffer_diff = Buffer.alloc(3 * numVoxels);
let buffer_diff_sao = Buffer.alloc(3 * numVoxels);
for(let i = 0; i < numVoxels; i++){

	let r = buffer.readUint8(3 * numVoxels + 3 * i + 0);
	let g = buffer.readUint8(3 * numVoxels + 3 * i + 1);
	let b = buffer.readUint8(3 * numVoxels + 3 * i + 2);

	if(i === 0){
		buffer_diff.writeUInt8(r, 3 * i + 0);
		buffer_diff.writeUInt8(g, 3 * i + 1);
		buffer_diff.writeUInt8(b, 3 * i + 2);

		buffer_diff_sao.writeUInt8(r, 0 * numVoxels + i);
		buffer_diff_sao.writeUInt8(g, 1 * numVoxels + i);
		buffer_diff_sao.writeUInt8(b, 2 * numVoxels + i);
		
	}else{
		// previous rgb
		let rp = buffer.readUint8(3 * numVoxels + 3 * i + 0 - 3);
		let gp = buffer.readUint8(3 * numVoxels + 3 * i + 1 - 3);
		let bp = buffer.readUint8(3 * numVoxels + 3 * i + 2 - 3);

		let diff_r = (r - rp + 256) % 256;
		let diff_g = (g - gp + 256) % 256;
		let diff_b = (b - bp + 256) % 256;

		buffer_diff.writeUInt8(diff_r, 3 * i + 0);
		buffer_diff.writeUInt8(diff_g, 3 * i + 1);
		buffer_diff.writeUInt8(diff_b, 3 * i + 2);

		buffer_diff_sao.writeUInt8(r, 0 * numVoxels + i);
		buffer_diff_sao.writeUInt8(g, 1 * numVoxels + i);
		buffer_diff_sao.writeUInt8(b, 2 * numVoxels + i);
	}
}

let buffer_r = Buffer.alloc(numVoxels);
for(let i = 0; i < numVoxels; i++){
	let r = buffer.readUint8(3 * numVoxels + 3 * i + 0);

	buffer_r.writeUInt8(r, i);
}

fsp.writeFile("./buffer_interleaved_u8.bin", buffer_interleaved_u8);
fsp.writeFile("./buffer_soa_u8.bin", buffer_soa_u8);
fsp.writeFile("./buffer_diff.bin", buffer_diff);
fsp.writeFile("./buffer_diff_sao.bin", buffer_diff_sao);
fsp.writeFile("./buffer_r.bin", buffer_r);



for(let i = 0; i < 16; i++){
	let r = buffer.readUint8(3 * numVoxels + 3 * i + 0);
	let g = buffer.readUint8(3 * numVoxels + 3 * i + 1);
	let b = buffer.readUint8(3 * numVoxels + 3 * i + 2);

	console.log({r, g, b});
}