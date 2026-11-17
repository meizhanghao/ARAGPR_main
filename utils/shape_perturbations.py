"""Small, reproducible image edits using existing binary region masks.

These edits are sensitivity probes, not certified label-preserving medical
counterfactuals. A geometric edit moves image pixels and every retained mask
together. Appearance edits leave masks unchanged and preserve their boundaries.
"""
import cv2
import numpy as np


def clean_binary(raw, kernel=3, minimum_area=100):
    if raw.ndim != 2 or not np.isin(raw, [0,255]).all():
        raise ValueError("Expected an actual binary mask.")
    median = cv2.medianBlur(raw, kernel)
    closed = cv2.morphologyEx(median,cv2.MORPH_CLOSE,np.ones((kernel,kernel),np.uint8))
    contours,_ = cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    result=np.zeros_like(raw)
    for contour in contours:
        if cv2.contourArea(contour)>minimum_area:
            cv2.drawContours(result,[contour],0,255,-1)
    return result


def recover_crop_position(image, crop, min_score=.98, min_gap=.005, max_mae=6.0):
    """Recover legacy crop coordinates from its unchanged RGB parent image.

    New manifests should supply bbox coordinates directly. Legacy coordinates
    are accepted only with a strong, unique match and small reconstruction
    error; otherwise the sample is excluded with a recorded reason.
    """
    h,w=crop.shape[:2]
    if h>image.shape[0] or w>image.shape[1] or min(h,w)<3 or float(crop.std())<2:
        raise ValueError("Degenerate crop or crop larger than original.")
    response=cv2.matchTemplate(image,crop,cv2.TM_CCOEFF_NORMED)
    _,score,_,(x,y)=cv2.minMaxLoc(response)
    mae=float(np.abs(image[y:y+h,x:x+w].astype(float)-crop).mean())
    response[max(0,y-3):y+4,max(0,x-3):x+4]=-2
    second=float(response.max())
    if not np.isfinite(score) or score<min_score or score-second<min_gap or mae>max_mae:
        raise ValueError(f"Uncertain legacy crop registration: score={score:.5f}, gap={score-second:.5f}, MAE={mae:.3f}")
    return (x,y,w,h),{"match_score":float(score),"alternative_score":second,
                       "match_gap":float(score-second),"crop_mae":mae}


def local_warp_maps(mask, strength):
    """Compact smooth anisotropic deformation centered on the dominant region.

    The inverse displacement vanishes outside an ellipse around that region.
    Positive strength stretches horizontally and compresses vertically. It is
    approximately area-preserving near the center, not exactly so everywhere.
    A positive Jacobian is required to prevent folded sampling maps.
    """
    yy,xx=np.indices(mask.shape,dtype=np.float32)
    points=cv2.findNonZero(mask)
    if points is None: raise ValueError("Empty dominant region.")
    x,y,w,h=cv2.boundingRect(points)
    cx,cy=x+(w-1)/2,y+(h-1)/2
    rx,ry=.85*w+16,.85*h+16
    radius=np.sqrt(((xx-cx)/rx)**2+((yy-cy)/ry)**2)
    window=np.where(radius<1,np.cos(np.minimum(radius,1)*np.pi/2)**2,0).astype(np.float32)
    mx=xx-strength*(xx-cx)*window
    my=yy+strength*(yy-cy)*window
    dx_y,dx_x=np.gradient(mx)
    dy_y,dy_x=np.gradient(my)
    minimum=float((dx_x*dy_y-dx_y*dy_x).min())
    if minimum<=.25: raise ValueError("Warp would fold or excessively compress the image.")
    return mx,my,minimum


def warp_image_and_masks(image,masks,strength):
    dominant=max(masks,key=np.count_nonzero)
    mx,my,jacobian=local_warp_maps(dominant,strength)
    rgb=cv2.remap(image,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    regions=[cv2.remap(mask,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,
                       borderValue=0) for mask in masks]
    return rgb,regions,jacobian


def appearance_blur(image,masks,sigma_at_224,scope):
    """Blur texture inside eroded regions or outside dilated regions.

    A boundary band remains byte-identical to the original; no image-mask
    boundary is moved. 'Outside' means outside the retained SAM regions and
    must not be interpreted as an expert-annotated clinical background.
    """
    union=np.maximum.reduce(masks)
    radius=max(2,int(round(3*min(image.shape[:2])/224)))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    if scope=="inside":
        editable=cv2.erode(union,kernel)>0
    elif scope=="outside":
        editable=cv2.dilate(union,kernel)==0
    else: raise ValueError(scope)
    sigma=sigma_at_224*min(image.shape[:2])/224
    blurred=cv2.GaussianBlur(image,(0,0),sigmaX=sigma,sigmaY=sigma)
    result=image.copy()
    result[editable]=blurred[editable]
    # Returning copies prevents an accidental in-place mask mutation.
    return result,[mask.copy() for mask in masks],float(editable.mean())
