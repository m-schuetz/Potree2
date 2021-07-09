import {LasLoader, Header} from "LasLoader";
import {Vector3} from "potree";

export function splitLasfile(las){
	let maxBatchSize = 10_000;

	let batches = [];
	let batch = null;
	for(let i = 0; i < las.header.numPoints; i++){

		if(batch == null || batch.header.numPoints >= maxBatchSize){
			// new batch

			let newBatchSize = Math.min(las.header.numPoints - i, maxBatchSize);
			let header = new Header();
			header.numPoints = 0;
			header.scale = las.header.scale.clone();
			header.offset = las.header.scale.clone();
			header.min = new Vector3(Infinity, Infinity, Infinity);
			header.max = new Vector3(-Infinity, -Infinity, -Infinity);

			let positionf32 = new Float32Array(3 * newBatchSize);
			let color = new Uint8Array(4 * newBatchSize);
			let buffers = {positionf32, color};

			batch = {header, buffers};

			batches.push(batch);
		}

		let j = batch.header.numPoints;

		let x = las.buffers.positionf32[3 * i + 0]
		let y = las.buffers.positionf32[3 * i + 1]
		let z = las.buffers.positionf32[3 * i + 2]

		batch.header.min.x = Math.min(batch.header.min.x, x);
		batch.header.min.y = Math.min(batch.header.min.y, y);
		batch.header.min.z = Math.min(batch.header.min.z, z);
		batch.header.max.x = Math.max(batch.header.max.x, x);
		batch.header.max.y = Math.max(batch.header.max.y, y);
		batch.header.max.z = Math.max(batch.header.max.z, z);

		batch.buffers.positionf32[3 * j + 0] = x;
		batch.buffers.positionf32[3 * j + 1] = y;
		batch.buffers.positionf32[3 * j + 2] = z;

		batch.buffers.color[4 * j + 0] = las.buffers.color[4 * i + 0];
		batch.buffers.color[4 * j + 1] = las.buffers.color[4 * i + 1];
		batch.buffers.color[4 * j + 2] = las.buffers.color[4 * i + 2];
		batch.buffers.color[4 * j + 3] = las.buffers.color[4 * i + 3];

		batch.header.numPoints++;
	}

	return batches;
}


export function randomSelection(las){
	let sampleSize = 100_000;

	let header = new Header();
	header.numPoints = sampleSize;
	header.scale = las.header.scale.clone();
	header.offset = las.header.scale.clone();
	header.min = las.header.min.clone();
	header.max = las.header.max.clone();

	let positionf32 = new Float32Array(3 * sampleSize);
	let color = new Uint8Array(4 * sampleSize);
	let buffers = {positionf32, color};

	let batch = {header, buffers};
	
	for(let i = 0; i < sampleSize; i++){

		let sourceIndex = Math.floor(Math.random() * las.header.numPoints);

		let x = las.buffers.positionf32[3 * sourceIndex + 0]
		let y = las.buffers.positionf32[3 * sourceIndex + 1]
		let z = las.buffers.positionf32[3 * sourceIndex + 2]

		batch.buffers.positionf32[3 * i + 0] = x;
		batch.buffers.positionf32[3 * i + 1] = y;
		batch.buffers.positionf32[3 * i + 2] = z;

		batch.buffers.color[4 * i + 0] = las.buffers.color[4 * sourceIndex + 0];
		batch.buffers.color[4 * i + 1] = las.buffers.color[4 * sourceIndex + 1];
		batch.buffers.color[4 * i + 2] = las.buffers.color[4 * sourceIndex + 2];
		batch.buffers.color[4 * i + 3] = las.buffers.color[4 * sourceIndex + 3];
	}

	return batch;
}