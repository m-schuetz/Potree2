
import {SceneNode} from "../../scene/SceneNode.js";



export class Mesh extends SceneNode{

	constructor(name, buffers, numVertices){
		super(name);

		this.buffers = buffers;
		this.numVertices = numVertices;
	}

}

