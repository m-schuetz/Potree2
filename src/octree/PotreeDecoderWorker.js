
import {toWebgpuAttribute, webgpuTypedArrayName, createAttributeReader, createWebgpuWriter} from "./PointAttributes.js"

// function buildComposite(attributes, numPoints, source){
// 	let compositeSize = 0;
// 	for(let attribute of attributes){
// 		let alignedSize =  4 * (1 + parseInt(attribute.byteSize / 4));

// 		compositeSize += alignedSize;
// 	}
// 	let compositeBuffer = new ArrayBuffer(compositeSize);


// 	for(let attribute of attributes){
// 		//let alignedSize =  4 * (1 + parseInt(attribute.byteSize / 4));

// 		for(let i = 0; i < numPoints; i++){
// 			for(let j = 0; j < attribute.numElements; j++){
// 				let sourceOffset = i * sourcePointSize + sourceAttributeOffset + j * sourceElementSize;
// 				let targetOffset = i * targetAttributeSize + j * targetElementSize;

// 				let value = read(sourceOffset);
// 				write(targetOffset, value);
// 			}
// 		}

// 		sourceAttributeOffset += attribute.byteSize;
// 	}

// 	let composite = {
// 		buffer: compositeBuffer,
// 	};

// 	return composite;
// }

onmessage = function (event) {

	let tStart = performance.now();
	
	let {buffer, attributes} = event.data;

	let bytesPerPoint = attributes.reduce((a, v) => a + v.byteSize, 0);
	let numPoints = buffer.byteLength / bytesPerPoint;
	let source = new DataView(buffer);
	
	let attributeBuffers = [];
	let sourceAttributeOffset = 0;

	// let composite = buildComposite(attributes, numPoints, source);
	
	
	// SEPERATE BUFFERS
	for (let attribute of attributes) {

		let webgpuAttribute = toWebgpuAttribute(attribute);
		
		let sourcePointSize = bytesPerPoint;
		let sourceAttributeSize = attribute.byteSize;
		let sourceElementSize = attribute.byteSize / attribute.numElements;

		let targetAttributeSize = webgpuAttribute.byteSize;
		let targetElementSize = webgpuAttribute.byteSize / webgpuAttribute.numElements;

		let targetBuffer = new ArrayBuffer(numPoints * targetAttributeSize);
		let target = new DataView(targetBuffer);

		let read = createAttributeReader(attribute, source);
		let write = createWebgpuWriter(webgpuAttribute.type, target);

		for(let i = 0; i < numPoints; i++){
			for(let j = 0; j < attribute.numElements; j++){
				let sourceOffset = i * sourcePointSize + sourceAttributeOffset + j * sourceElementSize;
				let targetOffset = i * targetAttributeSize + j * targetElementSize;

				let value = read(sourceOffset);
				write(targetOffset, value);
			}
		}

		attributeBuffers.push({name: attribute.name, array: targetBuffer});
	
		sourceAttributeOffset += attribute.byteSize;
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
