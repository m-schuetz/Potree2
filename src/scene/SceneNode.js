
import {Vector3} from "../math/Vector3.js";
import {Quaternion} from "../math/Quaternion.js";
import {Matrix4} from "../math/Matrix4.js";

export class SceneNode{

	constructor(name){
		this.name = name ?? "";
		this.position = new Vector3(0, 0, 0);
		this.scale = new Vector3(1, 1, 1);
		this.orientation = new Quaternion(0, 0, 0, 1);
		this.children = [];
	}

	add(node){
		this.children.push(node);
	}

}