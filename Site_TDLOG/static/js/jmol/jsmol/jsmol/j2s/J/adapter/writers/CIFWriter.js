Clazz.declarePackage ("J.adapter.writers");
Clazz.load (["J.api.JmolWriter"], "J.adapter.writers.CIFWriter", ["JU.P3", "$.PT", "$.SB", "JV.Viewer"], function () {
c$ = Clazz.decorateAsClass (function () {
this.vwr = null;
this.oc = null;
Clazz.instantialize (this, arguments);
}, J.adapter.writers, "CIFWriter", null, J.api.JmolWriter);
Clazz.makeConstructor (c$, 
function () {
});
Clazz.overrideMethod (c$, "set", 
function (viewer, oc, data) {
this.vwr = viewer;
this.oc = (oc == null ? this.vwr.getOutputChannel (null, null) : oc);
}, "JV.Viewer,JU.OC,~A");
Clazz.overrideMethod (c$, "write", 
function (bs) {
if (bs == null) bs = this.vwr.bsA ();
try {
var sb =  new JU.SB ();
sb.append ("# primitive CIF file created by Jmol " + JV.Viewer.getJmolVersion () + "\ndata_primitive");
var uc = this.vwr.ms.getUnitCellForAtom (bs.nextSetBit (0));
var params = (uc == null ?  Clazz.newFloatArray (-1, [1, 1, 1, 90, 90, 90]) : uc.getUnitCellAsArray (false));
sb.append ("\n_cell_length_a ").appendF (params[0]);
sb.append ("\n_cell_length_b ").appendF (params[1]);
sb.append ("\n_cell_length_c ").appendF (params[2]);
sb.append ("\n_cell_angle_alpha ").appendF (params[3]);
sb.append ("\n_cell_angle_beta ").appendF (params[4]);
sb.append ("\n_cell_angle_gamma ").appendF (params[5]);
sb.append ("\n\n_symmetry_space_group_name_H-M 'P 1'\nloop_\n_space_group_symop_operation_xyz\n'x,y,z'");
sb.append ("\n\nloop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n");
var atoms = this.vwr.ms.at;
var p =  new JU.P3 ();
for (var i = bs.nextSetBit (0); i >= 0; i = bs.nextSetBit (i + 1)) {
var a = atoms[i];
p.setT (a);
if (uc != null) uc.toFractional (p, false);
sb.append (a.getElementSymbol ()).append ("\t").append (JU.PT.formatF (p.x, 10, 5, true, false)).append ("\t").append (JU.PT.formatF (p.y, 10, 5, true, false)).append ("\t").append (JU.PT.formatF (p.z, 10, 5, true, false)).append ("\n");
}
this.oc.append (sb.toString ());
} catch (e) {
if (Clazz.exceptionOf (e, Exception)) {
} else {
throw e;
}
}
return this.toString ();
}, "JU.BS");
Clazz.overrideMethod (c$, "toString", 
function () {
return (this.oc == null ? "" : this.oc.toString ());
});
});
