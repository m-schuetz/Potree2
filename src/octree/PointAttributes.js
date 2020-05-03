

export class PointAttribute{
	
	constructor(name, type, numElements){
		this.name = name;
		this.type = type;
		this.numElements = numElements;
		this.byteSize = numElements * type.size;
		this.description = "";
		this.range = [Infinity, -Infinity];
	}

};

