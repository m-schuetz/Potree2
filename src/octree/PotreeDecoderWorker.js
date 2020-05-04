

onmessage = function (event) {

	let tStart = performance.now();
	
	let {buffer, attributes} = event.data;

	let bytesPerPoint = attributes.reduce((a, v) => a + v.byteSize, 0);
	let numPoints = buffer.byteLength / bytesPerPoint;
	let cv = new DataView(buffer);
	
	let attributeBuffers = [];
	let attributeOffset = 0;

	for (let attribute of attributes) {
		
		if(attribute.name === "rgb"){
			let buff = new ArrayBuffer(numPoints * 4);
			let colors = new Uint8Array(buff);

			for (let j = 0; j < numPoints; j++) {
				let pointOffset = j * bytesPerPoint;

				let r = cv.getUint16(pointOffset + attributeOffset + 0);
				let g = cv.getUint16(pointOffset + attributeOffset + 2);
				let b = cv.getUint16(pointOffset + attributeOffset + 4);

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
					let value = cv.getUint8(pointOffset + attributeOffset + k);
					uint8[j * attributeSize + k] = value;
				}
			}

			//attributeBuffers[attribute.name] = { buffer: buff, attribute: attribute };
			attributeBuffers.push({name: attribute.name, array: buff});
		}
		
		attributeOffset += attribute.byteSize;
	}

	let message = {
		buffer: buffer,
		attributeBuffers: attributeBuffers,
	};

	let transferables = [];
	// for (let property in message.attributeBuffers) {
	// 	transferables.push(message.attributeBuffers[property].buffer);
	// }
	for(let buffer of message.attributeBuffers){
		transferables.push(buffer.array);
	}
	transferables.push(buffer);


	let duration = performance.now() - tStart;
	let pointsPerSec = ((numPoints / duration) * 1000);
	pointsPerSec = (pointsPerSec / (1000 * 1000)).toFixed(1);
	console.log(`${name}: ${duration.toFixed(3)}ms, numPoints: ${numPoints}, points/sec: ${pointsPerSec}M`);


	postMessage(message, transferables);
};
