
import { render as renderNormal, NormalMaterial } from "./NormalMaterial.js";
import { render as renderPhong, PhongMaterial } from "./PhongMaterial.js";
import { render as renderWireframe, WireframeMaterial } from "./WireframeMaterial.js";

export function renderMeshes(meshes, drawstate){

	for(let mesh of meshes){
		if(mesh.material instanceof NormalMaterial){
			renderNormal(mesh, drawstate);
		}else if(mesh.material instanceof PhongMaterial){
			renderPhong(mesh, drawstate);
		}else if(mesh.material instanceof WireframeMaterial){
			renderWireframe(mesh, drawstate);
		}
	}

}