
import { render as renderNormal, NormalMaterial } from "./NormalMaterial.js";
import { render as renderPhong, PhongMaterial } from "./PhongMaterial.js";
import * as Timer from "../../renderer/Timer.js";


export function render(renderer, pass, node, camera, renderables){

	// Timer.timestamp(pass.passEncoder, "meshes-start");

	if(node.material instanceof NormalMaterial){
		renderNormal(renderer, pass, node, camera, renderables);
	}else if(node.material instanceof PhongMaterial){
		renderPhong(renderer, pass, node, camera, renderables);
	}

	// Timer.timestamp(pass.passEncoder, "meshes-end");

}