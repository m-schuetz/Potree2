
import {promises as fsp} from "fs";

let plyDir = "D:/temp/wetransfer-05965b/Additive-Heidentor-Non-Averaged";

class Node{

	constructor(name){
		this.name = name;
		this.children = new Array(8).fill(null);
		this.byteOffset = 0;
		this.byteSize = 0;
	}

}

function buildHierarchy(files){

	files.sort((a, b) => {
		if(a.length !== b.length){
			return a.length - b.length;
		}else{
			return 0;
		}
	});

	let nodes = [];
	let root = new Node("r");
	let map = new Map();
	map.set("r", root);
	nodes.push(root);

	for(let i = 1; i < files.length; i++){
		let file = files[i];
		let name = file.substring(0, file.length - 4);
		let parentName = name.substring(0, name.length - 1);

		let childIndex = parseInt(name[name.length - 1]);
		let parent = map.get(parentName);
		let child = new Node(name);

		parent.children[childIndex] = child;
		map.set(name, child);
		nodes.push(child);
	}

	return {root, nodes};
}

async function createPointFile(nodes){

	let endHeader = [

	];

	for(let node of nodes){
		let path = `${plyDir}/${node.name}.ply`;

		let buffer = await fsp.readFile(path);

		TODO

		break;

	}

}

async function convert(){

	let listing = await fsp.readdir(plyDir);
	let plyFiles = listing.filter(file => file.endsWith(".ply"));

	let {root, nodes} = buildHierarchy(plyFiles);

	await createPointFile(nodes);


}



convert();
