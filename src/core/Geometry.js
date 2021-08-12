
import {Box3} from "potree";

export class Geometry{

	constructor({buffers, indices, numElements} = {}){
		this.buffers = buffers ?? [];
		this.indices = indices ?? null;
		this.numElements = numElements ?? 0;
		this.boundingBox = new Box3();
	}

	findBuffer(name){
		return this.buffers.find(buffer => buffer.name === name)?.buffer;
	}

}