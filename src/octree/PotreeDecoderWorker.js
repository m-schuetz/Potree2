


// WebGPU vertex formats: https://gpuweb.github.io/gpuweb/#vertex-formats
// not all alignments are possible. 4 byte alignment seems preferable.
// e.g. if an attribute requires 1 byte, promote it to a 4 byte type
// It may be advantageous to think of packing multiply attributes in the future



class TypeInfo{
	constructor(potreeType, potreeNumElements, webgpuType, webgpuNumElements){
		this.potreeType = potreeType;
		this.potreeNumElements = potreeNumElements;
		this.webgpuNumElements = webgpuNumElements;
		this.webgpuType = webgpuType;
		this.typedArray = this.getTypedArrayType(webgpuType);
	}

	getTypedArrayType(webgpuType){

		if(webgpuType.includes("uchar")){
			return Uint8Array;
		}else if(webgpuType.includes("char")){
			return Int8Array;
		}else if(webgpuType.includes("ushort")){
			return Uint16Array;
		}else if(webgpuType.includes("short")){
			return Int16Array;
		}else if(webgpuType.includes("uint")){
			return Uint32Array;
		}else if(webgpuType.includes("int")){
			return Int32Array;
		}else if(webgpuType.includes("float")){
			return Float32Array;
		}

	}

	// getTypedArrayType(){
	// 	let typedArrayTypes = {
	// 		"int8": Int8Array,
	// 		"int16": Int16Array,
	// 		"int32": Int32Array,
	// 		"uint8": Uint8Array,
	// 		"uint16": Uint16Array,
	// 		"uint32": Uint32Array,
	// 		"float": Float32Array,
	// 		"double": Float64Array,
	// 	};

	// 	return typedArrayTypes[this.potreeType];
	// }
};

let dbg = 0;
function getTypeInfo(attribute){

	

	let {type, numElements, byteSize} = attribute;
	let typeSize = byteSize / numElements;

	let typeInfo = null;

	if(type === "uint8"){
		if(numElements === 1){
			typeInfo = new TypeInfo(type, 1, "int", 1);
		}
	}


	if(dbg === 0){
		console.log(attribute);
		console.log(typeInfo);
	}
	dbg++;

	return typeInfo;
}

onmessage = function (event) {

	let tStart = performance.now();
	
	let {buffer, attributes} = event.data;

	let bytesPerPoint = attributes.reduce((a, v) => a + v.byteSize, 0);
	let numPoints = buffer.byteLength / bytesPerPoint;
	let view = new DataView(buffer);
	
	let attributeBuffers = [];
	let attributeOffset = 0;

	for (let attribute of attributes) {
		
		// if(false){
		// 	let bpe = typeInfo.typedArray.BYTES_PER_ELEMENT;
		// 	let bpp = pbe * typeInfo.webgpuNumElements;

		// 	let buff = new ArrayBuffer(numPoints * bpp);
		// 	let target = new typeInfo.typedArray(buff);

		// 	let read = reader(attribute, view);
		// 	let write = writer(attribute, target);

		// 	for(let i = 0; i < numPoints; i++){
		// 		for(let j = 0; j < attribute.numElements; j++){
		// 			let value = read(pointOffset + attributeOffset + j);
		// 			write(value, pos);
		// 		}
		// 	}

		// }else 
		if(attribute.name === "rgb"){
			let buff = new ArrayBuffer(numPoints * 4);
			let colors = new Uint8Array(buff);

			for (let j = 0; j < numPoints; j++) {
				let pointOffset = j * bytesPerPoint;

				let r = view.getUint16(pointOffset + attributeOffset + 0);
				let g = view.getUint16(pointOffset + attributeOffset + 2);
				let b = view.getUint16(pointOffset + attributeOffset + 4);

				r = r >= 256 ? r / 256 : r;
				g = g >= 256 ? g / 256 : g;
				b = b >= 256 ? b / 256 : b;

				colors[4 * j + 0] = r;
				colors[4 * j + 1] = g;
				colors[4 * j + 2] = b;
			}

			//attributeBuffers[attribute.name] = { buffer: buff, attribute: attribute };
			attributeBuffers.push({name: attribute.name, array: buff});
		}else{

			let attributeSize = attribute.byteSize;
			let buff = new ArrayBuffer(numPoints * attributeSize);
			let uint8 = new Uint8Array(buff);

			for (let j = 0; j < numPoints; j++) {
				let pointOffset = j * bytesPerPoint;

				for(let k = 0; k < attributeSize; k++){
					let value = view.getUint8(pointOffset + attributeOffset + k);
					uint8[j * attributeSize + k] = value;
				}
			}

			attributeBuffers.push({name: attribute.name, array: buff});
		}
		
		attributeOffset += attribute.byteSize;
	}

	let message = {
		buffer: buffer,
		attributeBuffers: attributeBuffers,
	};

	let transferables = [];
	for(let buffer of message.attributeBuffers){
		transferables.push(buffer.array);
	}
	transferables.push(buffer);


	// let duration = performance.now() - tStart;
	// let pointsPerSec = ((numPoints / duration) * 1000);
	// pointsPerSec = (pointsPerSec / (1000 * 1000)).toFixed(1);
	// console.log(`${name}: ${duration.toFixed(3)}ms, numPoints: ${numPoints}, points/sec: ${pointsPerSec}M`);


	postMessage(message, transferables);
};
