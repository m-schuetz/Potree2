
import {Box3} from "potree";

let g_counter = 0;

export class Geometry{

	constructor({buffers, indices, numElements} = {}){
		this.buffers = buffers ?? [];
		this.indices = indices ?? null;
		this.numElements = numElements ?? 0;
		this.boundingBox = new Box3();

		this.id = g_counter;
		g_counter++;
	}

	findBuffer(name){
		return this.buffers.find(buffer => buffer.name === name)?.buffer;
	}

}