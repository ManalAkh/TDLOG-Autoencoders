Clazz.declarePackage ("J.adapter.writers");
Clazz.load (["J.api.JmolWriter"], "J.adapter.writers.PWMWriter", ["JU.P3", "$.PT"], function () {
c$ = Clazz.decorateAsClass (function () {
this.vwr = null;
this.oc = null;
this.uc = null;
Clazz.instantialize (this, arguments);
}, J.adapter.writers, "PWMWriter", null, J.api.JmolWriter);
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
var n = bs.cardinality ();
var line = JU.PT.formatStringI ("%12i\n", "i", n);
this.oc.append (line);
this.uc = this.vwr.ms.getUnitCellForAtom (bs.nextSetBit (0));
this.writeLattice ();
this.writePositions (bs);
} catch (e) {
if (Clazz.exceptionOf (e, Exception)) {
} else {
throw e;
}
}
return this.toString ();
}, "JU.BS");
Clazz.defineMethod (c$, "writeLattice", 
 function () {
this.oc.append ("Lattice vector\n");
if (this.uc == null) {
this.uc = this.vwr.getSymTemp ();
var bb = this.vwr.getBoundBoxCornerVector ();
var len = Math.round (bb.length () * 2);
this.uc.setUnitCell ( Clazz.newFloatArray (-1, [len, len, len, 90, 90, 90]), false);
}var abc = this.uc.getUnitCellVectors ();
var f = "%18.10p%18.10p%18.10p\n";
this.oc.append (JU.PT.sprintf (f, "p",  Clazz.newArray (-1, [abc[1]])));
this.oc.append (JU.PT.sprintf (f, "p",  Clazz.newArray (-1, [abc[2]])));
this.oc.append (JU.PT.sprintf (f, "p",  Clazz.newArray (-1, [abc[3]])));
});
Clazz.defineMethod (c$, "writePositions", 
 function (bs) {
this.oc.append ("Position, move_x, move_y, move_z\n");
var f = "%4i%18.12p%18.12p%18.12p  1  1  1\n";
var a = this.vwr.ms.at;
var p =  new JU.P3 ();
for (var i = bs.nextSetBit (0); i >= 0; i = bs.nextSetBit (i + 1)) {
p.setT (a[i]);
this.uc.toFractional (p, true);
this.oc.append (JU.PT.sprintf (f, "ip",  Clazz.newArray (-1, [Integer.$valueOf (a[i].getElementNumber ()), p])));
}
}, "JU.BS");
Clazz.overrideMethod (c$, "toString", 
function () {
return (this.oc == null ? "" : this.oc.toString ());
});
});
