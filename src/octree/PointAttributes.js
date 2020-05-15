

export class PointAttribute{
	
	constructor(name, type, numElements){
		this.name = name;
		this.type = type;
		this.numElements = numElements;
		this.byteSize = null;
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



export function webgpuTypedArrayName(typename){
	if(typename.includes("uchar")){
		return Uint8Array;
	}else if(typename.includes("char")){
		return Int8Array;
	}else if(typename.includes("ushort")){
		return Uint16Array;
	}else if(typename.includes("short")){
		return Int16Array;
	}else if(typename.includes("uint")){
		return Uint32Array;
	}else if(typename.includes("int")){
		return Int32Array;
	}else if(typename.includes("float")){
		return Float32Array;
	}else if(typename.includes("double")){
		return Int32Array;
	}else{
		throw `unsupported vertex type: ${typename}`;
	}

}

// https://gpuweb.github.io/gpuweb/#vertex-formats
// preferably promote all attributes to a size with a multiple of 4 bytes
export function toWebgpuAttribute(attribute){

	let webgpu = new PointAttribute(attribute.name, null, null);

	if(attribute.type.includes("int8")){
		let sign = attribute.type.startsWith("u") ? "u" : "";

		if(attribute.numElements === 1){
			webgpu.type = `${sign}int`;
			webgpu.numElements = 1;
			webgpu.byteSize = 4;
		}else{
			webgpu.type = `${sign}char4`;
			webgpu.numElements = 4;
			webgpu.byteSize = 4;
		}
	}else if(attribute.type.includes("int16")){
		let sign = attribute.type.startsWith("u") ? "u" : "";

		if(attribute.numElements === 1){
			webgpu.type = `${sign}int`;
			webgpu.numElements = 1;
			webgpu.byteSize = 4;
		}else if(attribute.numElements === 2){
			webgpu.type = `${sign}short2`;
			webgpu.numElements = 2;
			webgpu.byteSize = 4;
		}else{
			webgpu.type = `${sign}short4`;
			webgpu.numElements = 4;
			webgpu.byteSize = 8;
		}
	}else if(attribute.type.includes("int32")){
		let sign = attribute.type.startsWith("u") ? "u" : "";

		webgpu.type = `${sign}int${attribute.numElements}`;
		webgpu.numElements = attribute.numElements;
		webgpu.byteSize = attribute.byteSize;
	}else if(attribute.type === "float"){
		webgpu.type = `float${attribute.numElements}`;
		webgpu.numElements = attribute.numElements;
		webgpu.byteSize = attribute.byteSize;
	}else if(attribute.type === "double"){

		if(attribute.numElements > 1){
			throw `attribute currently not supported. 
				type: ${attribute.type},
				numElements: ${attribute.numElements}`;
		}

		// not yet supported by webgpu, so just treat it as 2 integers for now
		webgpu.type = `int2`;
		webgpu.numElements = 2;
		webgpu.byteSize = 8;
	}else{
		throw `attribute currently not supported. 
				type: ${attribute.type},
				numElements: ${attribute.numElements}`;
	}

	return webgpu;

}

export function createAttributeReader(attribute, view){

	let readerNames = {
		"int8":   "getInt8",
		"int16":  "getInt16",
		"int32":  "getInt32",
		"uint8":  "getUint8",
		"uint16": "getUint16",
		"uint32": "getUint32",
		"float":  "getFloat32",
		"double": "getFloat64",
	};

	let readerName = readerNames[attribute.type];

	let readX = view[readerName].bind(view);
	
	return (offset) => readX(offset, true);
	
}

export function createWebgpuWriter(webgpuType, view){
	let XArrayName = webgpuTypedArrayName(webgpuType).name;

	let writerNames = {
		"Int8Array":     "setInt8",
		"Int16Array":    "setInt16",
		"Int32Array":    "setInt32",
		"Uint8Array":    "setUint8",
		"Uint16Array":   "setUint16",
		"Uint32Array":   "setUint32",
		"Float32Array":  "setFloat32",
		"Float64Array":  "setFloat64",
	};

	let writerName = writerNames[XArrayName];

	let writeX = view[writerName].bind(view);

	return (offset, value) => writeX(offset, value, true);
}


export function webgpuToGlsl(typename){
	let mapping = {
		"uchar2": "uvec2",
		"uchar4": "uvec4",
		"char2": "uvec2",
		"char4": "uvec4",
		"ushort2": "uvec2",
		"ushort4": "uvec4",
		"short2": "uvec2",
		"short4": "uvec4",
		"float": "float",
		"float2": "vec2",
		"float3": "vec3",
		"float4": "vec4",
		"uint": "uint",
		"uint2": "uvec2",
		"uint3": "uvec3",
		"uint4": "uvec4",
		"int": "int",
		"int2": "ivec2",
		"int3": "ivec3",
		"int4": "ivec4",
	};

	let glslType = mapping[typename];

	if(!glslType){
		throw `unsupported webgpu/glsl type: ${typename}`;
	}

	return glslType;
}


