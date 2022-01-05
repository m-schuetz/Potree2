
import {SceneNode} from "./SceneNode.js";

export class Scene{

	constructor(){

		this.root = new SceneNode("root");
		this.root.scene = this;

	}

	add(parent, node){
		parent.children.push(node);
	}

};