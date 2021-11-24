
import {SceneNode} from "./SceneNode.js";

export class Scene{

	constructor(){

		this.root = new SceneNode("root");

	}

	add(parent, node){
		parent.children.push(node);
	}

};