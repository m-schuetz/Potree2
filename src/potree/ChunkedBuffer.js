
const {floor} = Math;

class Chunk{

	constructor(offset, size){
		this.offset = offset;
		this.size = size;
	}

}

export class ChunkedBuffer{

	constructor(capacity, chunkSize, renderer){

		this.renderer = renderer;
		this.capacity = capacity;
		this.chunkSize = chunkSize;
		this.numChunks = floor(capacity / chunkSize);
		this.chunks = new Array(this.numChunks);
		this.chunkPointer = 0;
		this.gpuBuffer = renderer.device.createBuffer({
			size: capacity,
			usage: GPUBufferUsage.VERTEX 
				| GPUBufferUsage.STORAGE
				| GPUBufferUsage.COPY_SRC
				| GPUBufferUsage.COPY_DST
				| GPUBufferUsage.UNIFORM,
		});

		for(let i = 0; i < this.numChunks; i++){
			this.chunks[i] = new Chunk(i * chunkSize, chunkSize);
		}

	}

	acquire(numBytes){

		let numChunks = floor((numBytes + this.chunkSize - 1) / this.chunkSize);

		let chunks = [];
		for(let i = 0; i < numChunks; i++){

			let chunk = this.chunks[this.chunkPointer + i];

			chunks.push(chunk);
		}

		this.chunkPointer += numChunks;

		return chunks;
	}

	release(chunks){

		for(let i = 0; i < chunks.length; i++){
			this.chunkPointer--;
			this.chunks[this.chunkPointer] = chunks[i];
		}

	}

}

