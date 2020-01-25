
class LASHeader{

	constructor(){
		this.versionMajor = 0;
		this.versionMinor = 0;
		this.headerSize = 0;
		this.offsetToPointData = 0;
		this.pointDataFormat = 0;
		this.pointDataRecordLength = 0;
		this.numPoints = 0;
		this.scale = [1, 1, 1];
		this.offset = [0, 0, 0];
		this.min = [0, 0, 0];
		this.max = [0, 0, 0];
	}

	static fromBuffer(buffer){

		let view = new DataView(buffer);
		let header = new LASHeader();

		header.versionMajor = view.getUint8(24);
		header.versionMinor = view.getUint8(25);
		header.headerSize = view.getUint16(94, true);
		header.offsetToPointData = view.getUint32(96, true);
		header.pointDataFormat = view.getUint8(104);
		header.pointDataRecordLength = view.getUint16(105, true);
		
		if(header.versionMajor <= 1 && header.versionMinor < 4){
			header.numPoints = view.getUint32(107, true);
		}else{
			console.warn("LAS 1.4 64 bit number of point values not yet supported");
			header.numPoints = view.getUint32(247 + 4, true);
		}

		header.scale = [
			view.getFloat64(131, true),
			view.getFloat64(139, true),
			view.getFloat64(147, true),
		];

		header.min = [
			view.getFloat64(187, true),
			view.getFloat64(203, true),
			view.getFloat64(219, true),
		];

		header.max = [
			view.getFloat64(179, true),
			view.getFloat64(195, true),
			view.getFloat64(211, true),
		];

		return header;
	}

};

export class LASLoader{

	constructor(url){
		this.url = url;
	}

	async loadHeader(){
		let byteOffset = 0;
		let byteSize = 375;

		let response = await fetch(this.url, {
			headers: {
				'content-type': 'multipart/byteranges',
				'Range': `bytes=${byteOffset}-${byteOffset + byteSize}`,
			},
		});

		let buffer = await response.arrayBuffer();

		let header = LASHeader.fromBuffer(buffer);

		this.header = header;
	}

	async *loadBatches(){

		let tStart = performance.now();

		let pointsLoaded = 0;
		let header = this.header;
		let numPoints = header.numPoints;
		let bytesPerPoint = header.pointDataRecordLength;
		let batchSize = 100_000; 

		while(pointsLoaded < numPoints){

			let tBatchStart = performance.now();

			let pointsLeft = numPoints - pointsLoaded;
			let currentBatchSize = Math.min(batchSize, pointsLeft);
			let fetchStart = header.offsetToPointData + pointsLoaded * bytesPerPoint;
			let fetchEnd = fetchStart + currentBatchSize * bytesPerPoint;

			let response = await fetch(this.url, {
				headers: {
					'content-type': 'multipart/byteranges',
					'Range': `bytes=${fetchStart}-${fetchEnd}`,
				},
			});

			let buffer = await response.arrayBuffer();
			let view = new DataView(buffer);

			let tLoadEnd = performance.now();

			let [scaleX, scaleY, scaleZ] = header.scale;
			let [offsetX, offsetY, offsetZ] = header.offset;
			let [minX, minY, minZ] = header.min;

			let rgbOffset = {
				2: 20,
				3: 28,
			}[header.pointDataFormat];

			let tParseStart = performance.now();

			let positions = new Float32Array(3 * currentBatchSize);
			let colors = new Float32Array(4 * currentBatchSize);

			for(let i = 0; i < currentBatchSize; i++){

				let pointOffset = i * bytesPerPoint;
				
				let ux = view.getInt32(pointOffset + 0, true);
				let uy = view.getInt32(pointOffset + 4, true);
				let uz = view.getInt32(pointOffset + 8, true);

				let x = (ux * scaleX) + offsetX - minX;
				let y = (uy * scaleY) + offsetY - minY;
				let z = (uz * scaleZ) + offsetZ - minZ;

				positions[3 * i + 0] = x;
				positions[3 * i + 1] = y;
				positions[3 * i + 2] = z;

				let r = view.getUint16(pointOffset + rgbOffset + 0) / 256;
				let g = view.getUint16(pointOffset + rgbOffset + 2) / 256;
				let b = view.getUint16(pointOffset + rgbOffset + 4) / 256;

				colors[4 * i + 0] = r;
				colors[4 * i + 1] = g;
				colors[4 * i + 2] = b;
				colors[4 * i + 3] = 1;
			}

			let batch = {
				header: header,
				size: currentBatchSize,
				positions: positions, 
				colors: colors,
			};

			pointsLoaded += currentBatchSize;

			let tEnd = performance.now();

			{ // print some stats
				let durationLoad = (tLoadEnd - tBatchStart);
				let durationParse = (tEnd - tParseStart);
				let durationLoadString = parseInt(durationLoad) + "ms";
				let durationParseString = parseInt(durationParse) + "ms";
				let strPoints = `${currentBatchSize.toLocaleString()} points`;
				console.log(`batch(${strPoints}) loaded in ${durationLoadString}, parsed in ${durationParseString}`);
			}

			yield batch;
		}

		let duration = performance.now() - tStart;
		let durationStr = (duration / 1000).toFixed(3) + "s";
		console.log(`point cloud loaded in: ${durationStr}`);

		return;
	}

};

