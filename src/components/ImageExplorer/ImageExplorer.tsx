import React, { useEffect, useState } from 'react';
import { useProjectStore, generateImageId } from '../../store/projectStore';
import { useFollicleStore } from '../../store/follicleStore';
import { useSettingsStore } from '../../store/settingsStore';
import { ProjectImage } from '../../types';
import { AlertTriangle, Settings } from 'lucide-react';

// Generate thumbnail using OffscreenCanvas
async function generateThumbnail(imageBitmap: ImageBitmap, maxSize: number = 48): Promise<string> {
  const aspectRatio = imageBitmap.width / imageBitmap.height;
  let width = maxSize;
  let height = maxSize;

  if (aspectRatio > 1) {
    height = maxSize / aspectRatio;
  } else {
    width = maxSize * aspectRatio;
  }

  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  ctx.drawImage(imageBitmap, 0, 0, width, height);
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.7 });
  return URL.createObjectURL(blob);
}

interface ImageItemProps {
  image: ProjectImage;
  isActive: boolean;
  annotationCount: number;
  hasCustomSettings: boolean;
  onSelect: () => void;
  onRemove: () => void;
}

const ImageItem: React.FC<ImageItemProps> = ({ image, isActive, annotationCount, hasCustomSettings, onSelect, onRemove }) => {
  const [thumbnail, setThumbnail] = useState<string>('');
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    let mounted = true;
    generateThumbnail(image.imageBitmap).then(url => {
      if (mounted) setThumbnail(url);
    });
    return () => {
      mounted = false;
      if (thumbnail) URL.revokeObjectURL(thumbnail);
    };
  }, [image.imageBitmap]);

  const truncateFileName = (name: string, maxLength: number = 15) => {
    if (name.length <= maxLength) return name;
    const ext = name.slice(name.lastIndexOf('.'));
    const baseName = name.slice(0, name.lastIndexOf('.'));
    const truncatedBase = baseName.slice(0, maxLength - ext.length - 3);
    return `${truncatedBase}...${ext}`;
  };

  return (
    <div
      className={`image-item ${isActive ? 'active' : ''}`}
      onClick={onSelect}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="image-thumbnail">
        {thumbnail ? (
          <img src={thumbnail} alt={image.fileName} />
        ) : (
          <div className="thumbnail-placeholder" />
        )}
      </div>
      <div className="image-info">
        <span className="image-name" title={image.fileName}>
          {truncateFileName(image.fileName)}
          {hasCustomSettings && (
            <span className="custom-settings-indicator" title="Has custom detection settings">
              <Settings size={10} />
            </span>
          )}
        </span>
        <span className="annotation-count">
          {annotationCount} annotation{annotationCount !== 1 ? 's' : ''}
        </span>
      </div>
      {isHovered && (
        <button
          className="remove-image-btn"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          title="Remove image"
        >
          ×
        </button>
      )}
    </div>
  );
};

export const ImageExplorer: React.FC = () => {
  const images = useProjectStore(state => state.images);
  const imageOrder = useProjectStore(state => state.imageOrder);
  const activeImageId = useProjectStore(state => state.activeImageId);
  const addImage = useProjectStore(state => state.addImage);
  const removeImage = useProjectStore(state => state.removeImage);
  const setActiveImage = useProjectStore(state => state.setActiveImage);

  const follicles = useFollicleStore(state => state.follicles);
  const deleteFolliclesForImage = useFollicleStore(state => state.deleteFolliclesForImage);

  const imageSettingsOverrides = useSettingsStore(state => state.imageSettingsOverrides);

  // State for delete confirmation dialog
  const [deleteConfirm, setDeleteConfirm] = useState<{
    imageId: string;
    fileName: string;
    annotationCount: number;
  } | null>(null);

  // Get annotation counts per image
  const getAnnotationCount = (imageId: string) => {
    return follicles.filter(f => f.imageId === imageId).length;
  };

  const handleAddImage = async () => {
    const result = await window.electronAPI.openImageDialog();
    if (!result) return;

    const { fileName, data } = result;

    // Create object URL and ImageBitmap
    const blob = new Blob([data]);
    const imageSrc = URL.createObjectURL(blob);
    const imageBitmap = await createImageBitmap(blob, { imageOrientation: 'from-image' });

    const newImage: ProjectImage = {
      id: generateImageId(),
      fileName,
      width: imageBitmap.width,
      height: imageBitmap.height,
      imageData: data,
      imageBitmap,
      imageSrc,
      viewport: { offsetX: 0, offsetY: 0, scale: 1 },
      createdAt: Date.now(),
      sortOrder: imageOrder.length,
    };

    addImage(newImage);
  };

  // Show confirmation dialog before removing image
  const handleRemoveImage = (imageId: string) => {
    const image = images.get(imageId);
    if (!image) return;

    const annotationCount = getAnnotationCount(imageId);
    setDeleteConfirm({
      imageId,
      fileName: image.fileName,
      annotationCount,
    });
  };

  // Actually remove the image after confirmation
  const confirmRemoveImage = () => {
    if (!deleteConfirm) return;

    // Delete annotations for this image
    deleteFolliclesForImage(deleteConfirm.imageId);
    // Remove the image
    removeImage(deleteConfirm.imageId);
    // Close dialog
    setDeleteConfirm(null);
  };

  const cancelRemoveImage = () => {
    setDeleteConfirm(null);
  };

  // Don't render if only one image
  if (images.size <= 1) {
    return null;
  }

  return (
    <>
      <div className="image-explorer">
        <div className="explorer-header">
          <h3>Images</h3>
          <button className="add-image-btn" onClick={handleAddImage} title="Add image">
            +
          </button>
        </div>
        <div className="image-list">
          {imageOrder.map(imageId => {
            const image = images.get(imageId);
            if (!image) return null;

            return (
              <ImageItem
                key={imageId}
                image={image}
                isActive={imageId === activeImageId}
                annotationCount={getAnnotationCount(imageId)}
                hasCustomSettings={imageSettingsOverrides.has(imageId)}
                onSelect={() => setActiveImage(imageId)}
                onRemove={() => handleRemoveImage(imageId)}
              />
            );
          })}
        </div>
      </div>

      {/* Delete Confirmation Dialog */}
      {deleteConfirm && (
        <div className="delete-confirm-overlay" onClick={cancelRemoveImage}>
          <div className="delete-confirm-dialog" onClick={e => e.stopPropagation()}>
            <div className="delete-confirm-icon">
              <AlertTriangle size={32} />
            </div>
            <h3>Delete Image?</h3>
            <p className="delete-confirm-filename">{deleteConfirm.fileName}</p>
            {deleteConfirm.annotationCount > 0 && (
              <p className="delete-confirm-warning">
                This will also delete <strong>{deleteConfirm.annotationCount}</strong> annotation{deleteConfirm.annotationCount !== 1 ? 's' : ''}.
              </p>
            )}
            <div className="delete-confirm-actions">
              <button className="delete-confirm-cancel" onClick={cancelRemoveImage}>
                Cancel
              </button>
              <button className="delete-confirm-delete" onClick={confirmRemoveImage}>
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
