
import { Vector3 } from "../../Potree.js";
import {GaussianSplats} from "./GaussianSplats.js";

export class GSLoader{

	constructor(){
		this.numSplats = 0;
		this.numSplatsLoaded = 0;
		this.bytesPerSplat = 0;

		this.offset_position = 0;
		this.offset_color = 0;
		this.offset_harmonics = 0;
		this.offset_opacity = 0;
		this.offset_scale = 0;
		this.offset_rotation = 0;
		this.firstContentByte = 0;

	}

	static async load(url){

		let loader = new GSLoader();
		let splats = new GaussianSplats(url);

		{ // load metadata
			let response = await fetch(url, {headers : { "Range": "bytes=0-2000"}});
			let text = await response.text();

			let endHeaderLocation = text.indexOf("end_header");
			loader.firstContentByte = endHeaderLocation + 11;

			let lines = text.split("\n");

			let byteOffset = 0;
			for(let i = 0; i < lines.length; i++){
				let line = lines[i];
				let tokens = line.split(" ");

				if(tokens[0] === "element" && tokens[1] === "vertex"){
					loader.numSplats = Number(tokens[2]);
				}else if(tokens[0] === "property" && tokens[1] === "float" && tokens[2] == "x"){
					loader.offset_position = byteOffset;
				}else if(tokens[0] === "property" && tokens[1] === "float" && tokens[2] == "f_dc_0"){
					loader.offset_color = byteOffset;
				}else if(tokens[0] === "property" && tokens[1] === "float" && tokens[2] == "f_rest_0"){
					loader.offset_harmonics = byteOffset;
				}else if(tokens[0] === "property" && tokens[1] === "float" && tokens[2] == "opacity"){
					loader.offset_opacity = byteOffset;
				}else if(tokens[0] === "property" && tokens[1] === "float" && tokens[2] == "scale_0"){
					loader.offset_scale = byteOffset;
				}else if(tokens[0] === "property" && tokens[1] === "float" && tokens[2] == "rot_0"){
					loader.offset_rotation = byteOffset;
				}

				if(tokens[0] === "property" && tokens[1] === "float"){
					byteOffset += 4;
				}
			}

			loader.bytesPerSplat += byteOffset;
		}

		{ // load splat data
			let numSplats = Math.floor(loader.numSplats / 1);

			let positions = new ArrayBuffer(12 * numSplats);
			let color     = new ArrayBuffer(16 * numSplats);
			let rotation  = new ArrayBuffer(16 * numSplats);
			let scale     = new ArrayBuffer(12 * numSplats);

			let v_positions = new DataView(positions);
			let v_color     = new DataView(color);
			let v_rotation  = new DataView(rotation);
			let v_scale     = new DataView(scale);

			let numSplatsLoaded = 0;

			let BATCH_SIZE = 10_000;
			let loadBatch = async (firstIndex, count) => {
				let first = loader.firstContentByte + firstIndex * loader.bytesPerSplat;
				let last = first + count * loader.bytesPerSplat;
				let response = await fetch(url, {headers : { "Range": `bytes=${first}-${last}`}});
				let buffer = await response.arrayBuffer();
				let view = new DataView(buffer);

				// console.log(`loading splats ${firstIndex} to ${count}`);

				for(let splatIndex = 0; splatIndex < count; splatIndex++){

					let targetIndex = numSplatsLoaded;

					{ // POSITION
						let x = view.getFloat32(splatIndex * loader.bytesPerSplat + loader.offset_position + 0, true);
						let y = view.getFloat32(splatIndex * loader.bytesPerSplat + loader.offset_position + 4, true);
						let z = view.getFloat32(splatIndex * loader.bytesPerSplat + loader.offset_position + 8, true);

						v_positions.setFloat32(12 * targetIndex + 0, x, true);
						v_positions.setFloat32(12 * targetIndex + 4, y, true);
						v_positions.setFloat32(12 * targetIndex + 8, z, true);
					}

					{ // COLOR & OPACITY
						let R = view.getFloat32(splatIndex * loader.bytesPerSplat + loader.offset_color + 0, true);
						let G = view.getFloat32(splatIndex * loader.bytesPerSplat + loader.offset_color + 4, true);
						let B = view.getFloat32(splatIndex * loader.bytesPerSplat + loader.offset_color + 8, true);

						let clamp = v => Math.min(Math.max(v, 0), 1);
						let O = view.getFloat32(splatIndex * loader.bytesPerSplat + loader.offset_opacity, true);
						let opacity = (1.0 / (1.0 + Math.exp(-O)));

						let CO = 0.28209479177387814; // spherical harmonics coefficient
						let r = clamp(0.5 + CO * R, 0, 1);
						let g = clamp(0.5 + CO * G, 0, 1);
						let b = clamp(0.5 + CO * B, 0, 1);

						v_color.setFloat32(16 * targetIndex +  0, r, true);
						v_color.setFloat32(16 * targetIndex +  4, g, true);
						v_color.setFloat32(16 * targetIndex +  8, b, true);
						v_color.setFloat32(16 * targetIndex + 12, opacity, true);
					}

					{ // SCALE
						let sx = view.getFloat32(splatIndex * loader.bytesPerSplat + loader.offset_scale + 0, true);
						let sy = view.getFloat32(splatIndex * loader.bytesPerSplat + loader.offset_scale + 4, true);
						let sz = view.getFloat32(splatIndex * loader.bytesPerSplat + loader.offset_scale + 8, true);

						sx = Math.exp(sx);
						sy = Math.exp(sy);
						sz = Math.exp(sz);

						v_scale.setFloat32(12 * targetIndex + 0, sx, true);
						v_scale.setFloat32(12 * targetIndex + 4, sy, true);
						v_scale.setFloat32(12 * targetIndex + 8, sz, true);

						// if(splatIndex < 5){
						// 	console.log({sx, sy, sz});
						// }
					}

					{ // ROTATION
						let w = view.getFloat32(splatIndex * loader.bytesPerSplat + loader.offset_rotation +  0, true);
						let x = view.getFloat32(splatIndex * loader.bytesPerSplat + loader.offset_rotation +  4, true);
						let y = view.getFloat32(splatIndex * loader.bytesPerSplat + loader.offset_rotation +  8, true);
						let z = view.getFloat32(splatIndex * loader.bytesPerSplat + loader.offset_rotation + 12, true);

						let length = Math.sqrt(x * x + y * y + z * z + w * w);

						v_rotation.setFloat32(16 * targetIndex +  0, x / length, true);
						v_rotation.setFloat32(16 * targetIndex +  4, y / length, true);
						v_rotation.setFloat32(16 * targetIndex +  8, z / length, true);
						v_rotation.setFloat32(16 * targetIndex + 12, w / length, true);
					}

					numSplatsLoaded++;
				}

				splats.numSplatsLoaded = numSplatsLoaded;

				if(numSplatsLoaded < numSplats){
					let first = numSplatsLoaded;
					let count = Math.min(numSplats - numSplatsLoaded, BATCH_SIZE);
					loadBatch(numSplatsLoaded, count);
				}
			};

			loadBatch(0, BATCH_SIZE);

			splats.numSplats = numSplats;
			splats.numSplatsLoaded = numSplatsLoaded;
			splats.splatData = {positions, color, rotation, scale };
		}

		return splats;
	}

}