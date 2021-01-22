
import {promises as fsp} from "fs";


let path = "D:/dev/pointclouds/eclepens.las";
let targetPath = "D:/dev/pointclouds/eclepens_shuffled.las";

function randomInt(start, end){
	return Math.floor(Math.random() * (end - start + 1) + start);
}

function shuffleIndices(n){
	let indices = new Array(n).fill(0).map((v, i) => i);

	for(let i = 0; i < indices.length - 1; i++){

		let sourceIndex = randomInt(i, indices.length - 1);

		let tmp = indices[i];
		indices[i] = indices[sourceIndex];
		indices[sourceIndex] = tmp;
	}

	return indices;
}

async function run(){
	let buffer = await fsp.readFile(path);

	let versionMajor = buffer.readUInt8(24);
	let versionMinor = buffer.readUInt8(25);
	let offsetToPointData = buffer.readUInt32LE(96);
	let recordLength = buffer.readUInt16LE(105);
	let numPoints;
	if(versionMajor >= 1 && versionMinor >= 4){
		numPoints = Number(buffer.readBigUInt64LE(247));
	}else{
		numPoints = buffer.readUInt32LE(107);
	}

	let order = shuffleIndices(numPoints);


	let targetBuffer = Buffer.alloc(buffer.byteLength);
	let target = await fsp.open(targetPath, "w");

	buffer.copy(targetBuffer, 0, 0, offsetToPointData);

	for(let i = 0; i < numPoints; i++){
		let index = order[i];
		let sourceStart = offsetToPointData + index * recordLength;
		let sourceEnd = sourceStart + recordLength;
		let targetStart = offsetToPointData + i * recordLength;

		buffer.copy(targetBuffer, targetStart, sourceStart, sourceEnd);
	}

	await target.write(targetBuffer);

	// await target.write(buffer.slice(0, offsetToPointData));

	// for(let i = 0; i < numPoints; i++){
	// 	let index = order[i];
	// 	let start = offsetToPointData + index * recordLength;
	// 	let end = start + recordLength;
	// 	let slice = buffer.slice(start, end);
	// 	await target.write(slice);
	// }




	await target.close();

}

run();

