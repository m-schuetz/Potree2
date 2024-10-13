
import {promises as fsp} from "fs";
import JSON5 from './json5.mjs';

class Point{
	constructor(){
		this.x = 0;
		this.y = 0;
		this.z = 0;
		this.r = 0;
		this.g = 0;
		this.b = 0;
	}
}

let txtJson = (await fsp.readFile("./metadata.json")).toString();
let fr = (await fsp.readFile("./r.csv")).toString();
let fr0 = (await fsp.readFile("./r0.csv")).toString();
let json = JSON5.parse(txtJson);

let root = json.nodes.find(n => n.name === "r");

function parsePoints(csv){

	let lines = csv.split("\n");
	let points = [];

	for(let line of lines){

		if(line.length < 2) continue;

		let tokens = line.split(", ");

		let point = new Point();
		point.x = parseFloat(tokens[0]);
		point.y = parseFloat(tokens[1]);
		point.z = parseFloat(tokens[2]);
		point.r = parseInt(tokens[3]);
		point.g = parseInt(tokens[4]);
		point.b = parseInt(tokens[5]);

		points.push(point);
	}

	return points;
}

function clamp(value, min, max){
	if(value < min) return min;
	if(value > max) return max;

	return value;
}

let r_points = parsePoints(fr);
let r0_points = parsePoints(fr0);

let gridsize = 128;
let grid = new Uint32Array(gridsize ** 3).fill(4294967295);
let min = {
	x: root.min[0],
	y: root.min[1],
	z: root.min[2],
};
let max = {
	x: root.max[0],
	y: root.max[1],
	z: root.max[2],
};
let boxsize = {
	x: max.x - min.x,
	y: max.y - min.y,
	z: max.z - min.z,
};

// insert parents into grid
for(let i = 0; i < r_points.length; i++){

	let point = r_points[i];

	let ix = Math.floor(clamp(gridsize * (point.x - min.x) / boxsize.x, 0, gridsize - 1));
	let iy = Math.floor(clamp(gridsize * (point.y - min.y) / boxsize.y, 0, gridsize - 1));
	let iz = Math.floor(clamp(gridsize * (point.z - min.z) / boxsize.z, 0, gridsize - 1));

	let voxelIndex = ix + iy * gridsize + iz * gridsize * gridsize;

	if(grid[voxelIndex] === 4294967295){
		grid[voxelIndex] = i;
	}else{
		console.log(point);
		console.log(voxelIndex);
		console.error("error!");
		exit(123);
	}

}

// link child voxels to parents
for(let i = 0; i < r0_points.length; i++){

	let point = r0_points[i];

	let ix = Math.floor(clamp(gridsize * (point.x - min.x) / boxsize.x, 0, gridsize - 1));
	let iy = Math.floor(clamp(gridsize * (point.y - min.y) / boxsize.y, 0, gridsize - 1));
	let iz = Math.floor(clamp(gridsize * (point.z - min.z) / boxsize.z, 0, gridsize - 1));

	let voxelIndex = ix + iy * gridsize + iz * gridsize * gridsize;

	if(grid[voxelIndex] === 4294967295){
		console.log(point);
		console.log(voxelIndex);
		console.error("error!");
		exit(123);
	}else{
		point.parent = r_points[grid[voxelIndex]];
	}

}

let differences = [];
let histogram_r = new Map();
let histogram_g = new Map();
let histogram_b = new Map();
for(let i = 0; i < r0_points.length; i++){
	let point = r0_points[i];

	let diff = [
		point.r - point.parent.r,
		point.g - point.parent.g,
		point.b - point.parent.b,
	];

	// if(!histogram_r.has(diff[0])){
	// 	histogram_r.set(diff[0], 0);
	// }
	// if(!histogram_g.has(diff[1])){
	// 	histogram_g.set(diff[1], 0);
	// }
	// if(!histogram_b.has(diff[2])){
	// 	histogram_b.set(diff[2], 0);
	// }

	// histogram_r.set(diff[0], histogram_r.get(diff[0]) + 1);
	// histogram_g.set(diff[1], histogram_g.get(diff[1]) + 1);
	// histogram_b.set(diff[2], histogram_b.get(diff[2]) + 1);

	differences.push(diff);
}

console.log(differences);
console.log(differences.length);


let diffbuffer_i32 = new Int32Array(r_points.length * 3);
let diffbuffer_u8 = new Uint8Array(r_points.length * 3);
let diffbuffer_u8_l2 = new Uint8Array(r_points.length * 3);

let {log2, abs, pow, round, sign} = Math;

for(let i = 0; i < differences.length; i++){
	diffbuffer_i32[3 * i + 0] = differences[i][0];
	diffbuffer_i32[3 * i + 1] = differences[i][1];
	diffbuffer_i32[3 * i + 2] = differences[i][2];

	diffbuffer_u8[3 * i + 0] = (differences[i][0] + 255) / 2;
	diffbuffer_u8[3 * i + 1] = (differences[i][1] + 255) / 2;
	diffbuffer_u8[3 * i + 2] = (differences[i][2] + 255) / 2;

	let getMask = (value) => {
		if(value >= 0) return 0;

		return 1 << 7;
	};

	diffbuffer_u8_l2[3 * i + 0] = log2(abs(differences[i][0])) | getMask(differences[i][0]);
	diffbuffer_u8_l2[3 * i + 1] = log2(abs(differences[i][1])) | getMask(differences[i][1]);
	diffbuffer_u8_l2[3 * i + 2] = log2(abs(differences[i][2])) | getMask(differences[i][2]);
}


fsp.writeFile("./encoded_i32.bin", diffbuffer_i32);
fsp.writeFile("./encoded_u8.bin", diffbuffer_u8);
fsp.writeFile("./encoded_u8_l2.bin", diffbuffer_u8_l2);