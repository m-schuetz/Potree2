

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

export function getArrayType(typename){

	let types = {
		"int8": Int8Array,
		"int16": Int16Array,
		"int32": Int32Array,
		"uint8": Uint8Array,
		"uint16": Uint16Array,
		"uint32": Uint32Array,
		"float": Float32Array,
		"double": Float64Array,
	};

	return types[typename];
}

