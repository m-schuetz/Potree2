
import {Geometry} from "../core/Geometry.js";


export async function load(url){

	let response = await fetch(url);
	let buffer = await response.arrayBuffer();

	let numLines = 0;
	let numVertices = 0;
	let numTexCoords = 0;
	let numNormals = 0;
	let numFaces = 0;

	let CODE_NEWLINE = "\n".charCodeAt(0);
	let CODE_CR = "\r".charCodeAt(0);
	let CODE_V = "v".charCodeAt(0);
	let CODE_T = "t".charCodeAt(0);
	let CODE_N = "n".charCodeAt(0);
	let CODE_F = "f".charCodeAt(0);
	let CODE_SPACE = " ".charCodeAt(0);

	{
		let tStart = performance.now();
		let u8 = new Uint8Array(buffer);

		for(let i = 0; i < u8.length; i++){

			if(u8[i] === CODE_NEWLINE){
				numLines++;

				if(u8[i + 1] === CODE_CR){
					i++;
				}

				let c1 = u8[i + 1];

				if(c1 === CODE_V){
					
					let c2 = u8[i + 2];

					if(c2 === CODE_N){
						numNormals++;
					}else if(c2 === CODE_T){
						numTexCoords++;
					}else{
						numVertices++;
					}

				}else if(c1 === CODE_F){

					let numFaceVertices = 0;
					while(i < u8.length - 1 && u8[i + 1] !== CODE_NEWLINE){
						
						if(u8[i + 1] === CODE_SPACE){
							numFaceVertices++;
						}

						i++;
					}

					numFaces = numFaces + numFaceVertices - 2;
				}
			}
		}

		let duration = performance.now() - tStart;
		console.log("numLines: ", numLines);
		console.log("numVertices: ", numVertices);
		console.log("numTexCoords: ", numTexCoords);
		console.log("numNormals: ", numNormals);
		console.log("numFaces: ", numFaces);
		console.log("duration: " + duration  + "ms");

	}

	// numFaces = numFaces * 2;

	let obj_vertices = new Float32Array(3 * numVertices);
	let obj_texcoords = new Float32Array(2 * numTexCoords);
	let obj_normals = new Float32Array(3 * numNormals);

	let obj_vertices_read = 0;
	let obj_texcoords_read = 0;
	let obj_normals_read = 0;

	let target_vertices = new Float32Array(3 * 3 * numFaces);
	let target_texcoords = new Float32Array(3 * 2 * numFaces);
	let target_normals = new Float32Array(3 * 3 * numFaces);
	let target_colors = new Float32Array(3 * 3 * numFaces);
	let faces_written = 0;

	{
		let tStart = performance.now();
		let u8 = new Uint8Array(buffer);

		let lineStart = 0;

		let processLine = (line) => {

			let tokens = line.split(" ");

			if(tokens[0] === "v"){
				// VERTEX

				let x = parseFloat(tokens[1]);
				let y = parseFloat(tokens[2]);
				let z = parseFloat(tokens[3]);

				obj_vertices[3 * obj_vertices_read + 0] = x;
				obj_vertices[3 * obj_vertices_read + 1] = y;
				obj_vertices[3 * obj_vertices_read + 2] = z;
				obj_vertices_read++;
			}else if(tokens[0] === "vn"){
				// NORMAL

				let x = parseFloat(tokens[1]);
				let y = parseFloat(tokens[2]);
				let z = parseFloat(tokens[3]);

				obj_normals[3 * obj_normals_read + 0] = x;
				obj_normals[3 * obj_normals_read + 1] = y;
				obj_normals[3 * obj_normals_read + 2] = z;
				obj_normals_read++;
			}else if(tokens[0] === "vt"){
				// TEXCOORD

				let u = parseFloat(tokens[1]);
				let v = parseFloat(tokens[2]);

				obj_texcoords[2 * obj_texcoords_read + 0] = u;
				obj_texcoords[2 * obj_texcoords_read + 1] = v;
				obj_texcoords_read++;
			}else if(tokens[0] === "f"){
				// FACE

				let p0 = tokens[1].split("/").map(v => parseInt(v));
				let p1 = tokens[2].split("/").map(v => parseInt(v));

				// TODO support >3 vertices
				for(let i = 3; i < tokens.length; i++){
					let p2 = tokens[i].split("/").map(v => parseInt(v));

					target_vertices[9 * faces_written + 0] = obj_vertices[3 * (p0[0] - 1) + 0];
					target_vertices[9 * faces_written + 1] = obj_vertices[3 * (p0[0] - 1) + 1];
					target_vertices[9 * faces_written + 2] = obj_vertices[3 * (p0[0] - 1) + 2];

					target_vertices[9 * faces_written + 3] = obj_vertices[3 * (p1[0] - 1) + 0];
					target_vertices[9 * faces_written + 4] = obj_vertices[3 * (p1[0] - 1) + 1];
					target_vertices[9 * faces_written + 5] = obj_vertices[3 * (p1[0] - 1) + 2];

					target_vertices[9 * faces_written + 6] = obj_vertices[3 * (p2[0] - 1) + 0];
					target_vertices[9 * faces_written + 7] = obj_vertices[3 * (p2[0] - 1) + 1];
					target_vertices[9 * faces_written + 8] = obj_vertices[3 * (p2[0] - 1) + 2];

					if(p0.length >= 2 && !Number.isNaN(p0[1])){
						target_texcoords[6 * faces_written + 0] = obj_texcoords[2 * (p0[1] - 1) + 0];
						target_texcoords[6 * faces_written + 1] = obj_texcoords[2 * (p0[1] - 1) + 1];

						target_texcoords[6 * faces_written + 2] = obj_texcoords[2 * (p1[1] - 1) + 0];
						target_texcoords[6 * faces_written + 3] = obj_texcoords[2 * (p1[1] - 1) + 1];

						target_texcoords[6 * faces_written + 4] = obj_texcoords[2 * (p2[1] - 1) + 0];
						target_texcoords[6 * faces_written + 5] = obj_texcoords[2 * (p2[1] - 1) + 1];
					}

					if(p0.length >=3){
						target_normals[9 * faces_written + 0] = obj_normals[3 * (p0[2] - 1) + 0];
						target_normals[9 * faces_written + 1] = obj_normals[3 * (p0[2] - 1) + 1];
						target_normals[9 * faces_written + 2] = obj_normals[3 * (p0[2] - 1) + 2];

						target_normals[9 * faces_written + 3] = obj_normals[3 * (p1[2] - 1) + 0];
						target_normals[9 * faces_written + 4] = obj_normals[3 * (p1[2] - 1) + 1];
						target_normals[9 * faces_written + 5] = obj_normals[3 * (p1[2] - 1) + 2];

						target_normals[9 * faces_written + 6] = obj_normals[3 * (p2[2] - 1) + 0];
						target_normals[9 * faces_written + 7] = obj_normals[3 * (p2[2] - 1) + 1];
						target_normals[9 * faces_written + 8] = obj_normals[3 * (p2[2] - 1) + 2];
					}

					p1 = p2;

					faces_written++;
				}


				
			}

		};

		for(let i = 0; i < u8.length; i++){

			if(u8[i] === CODE_NEWLINE){
				
				let lineEnd = i;

				let dec = new TextDecoder("utf-8");
				let arr = new Uint8Array(buffer.slice(lineStart, i));
				let line = dec.decode(arr);

				processLine(line);

				lineStart = lineEnd + 1;
			}

		}

		console.log(obj_vertices);
		console.log(target_vertices);
		console.log(target_texcoords);
		console.log(target_normals);

		let geometry = new Geometry();
		geometry.numElements = numFaces * 3;
		geometry.buffers = [
			{name: "position", buffer: target_vertices},
			{name: "color", buffer: target_colors},
			{name: "uv", buffer: target_texcoords},
			{name: "normal", buffer: target_normals},
		];

		let duration = performance.now() - tStart;
		console.log("duration: " + duration  + "ms");

		return geometry;
	}




};



