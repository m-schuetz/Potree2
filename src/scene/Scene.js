
import {SceneNode} from "./SceneNode.js";

export class Scene{

	constructor(){
		this.root = new SceneNode("root");
	}

	update(timestamp, delta){
		this.root.update(timestamp, delta);
	}

}