
export class Geometry{

	constructor({buffers, indices, numElements} = {}){
		this.buffers = buffers ?? [];
		this.indices = indices ?? null;
		this.numElements = numElements ?? 0;
	}

}